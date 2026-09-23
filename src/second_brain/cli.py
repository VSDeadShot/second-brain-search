"""Command line entry point.

The installed `sbs` script runs `main()`, not `cli` directly, so the console
streams are switched to UTF-8 before any command writes. When stdout is piped
or redirected on Windows, Python falls back to the locale codepage (cp1252),
and 11 of the 51 indexed documents contain emoji that cp1252 cannot encode.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import click

from .answering import Answer, Citation, answer_question
from .config import Config, ConfigError, load_config
from .discovery import DiscoveredDoc, discover_documents
from .generation import GeminiGenerator, GenerationError, Generator
from .embedding import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIMENSIONS,
    EMBEDDING_MODEL,
    Embedder,
    EmbeddingError,
    GeminiEmbedder,
    estimate_embedding_seconds,
    gemini_cache_namespace,
    plan_batches,
)
from .embedding_cache import EmbeddingCache
from .pipeline import EmbeddingPlan, IndexReport, collect_chunks, embedding_plan, store_chunks
from .evaluation import EvalSuiteError, load_eval_suite, report_to_dict, run_eval
from .retrieval import (
    DEFAULT_ANSWER_K,
    DEFAULT_K,
    RetrievalError,
    RetrievedChunk,
    retrieve,
)
from .store import ChunkStore

SNIPPET_CHARS = 240
# How many of the closest passages a refusal shows, labelled as not an answer.
REFUSAL_PASSAGES = 3

# src/second_brain/cli.py -> repo root. Both gitignored.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = _REPO_ROOT / ".chroma"
DEFAULT_CACHE_PATH = _REPO_ROOT / "data" / "embedding_cache.sqlite"
# The suite is product data, not a test fixture: `sbs eval` reads it. Reports go
# under data/, which is gitignored.
DEFAULT_EVAL_SUITE = _REPO_ROOT / "eval" / "retrieval_v1.toml"
DEFAULT_EVAL_OUT_DIR = _REPO_ROOT / "data" / "eval"

EmbedderFactory = Callable[[Config], Embedder]
GeneratorFactory = Callable[[Config], Generator]


def force_utf8_streams(*streams: Any) -> None:
    """Re-encode text streams as UTF-8. Streams that can't be reconfigured - a
    test runner's capture buffer, a StringIO - are left as they are."""
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB"):
        if value < 1024 or unit == "MB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} MB"  # pragma: no cover


def _as_dict(doc: DiscoveredDoc) -> dict[str, object]:
    return {
        "path": str(doc.path),
        "project": doc.project,
        "project_root": str(doc.project_root),
        "rel_path": doc.rel_path,
        "size_bytes": doc.size_bytes,
        "mtime": doc.mtime,
    }


def _load_config_or_fail() -> Config:
    try:
        return load_config()
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc


def _gemini_embedder(config: Config) -> Embedder:
    return GeminiEmbedder(api_key=config.gemini_api_key)


def _project_lines(report: IndexReport) -> list[str]:
    ordered = sorted(report.chunks_per_project.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    width = max((len(name) for name, _ in ordered), default=0)
    return [f"  {name:<{width}}  {count:>5}" for name, count in ordered]


def _skipped_lines(report: IndexReport) -> list[str]:
    count = len(report.skipped)
    noun = "file" if count == 1 else "files"
    if not count:
        return [f"Skipped 0 {noun}."]
    return [f"Skipped {count} {noun}:"] + [f"  {path}: {reason}" for path, reason in report.skipped]


def format_index_report(report: IndexReport, *, elapsed: float, index_dir: Path) -> str:
    lines = [
        f"Indexed {report.documents} documents across {report.projects} projects "
        f"-> {report.chunks} chunks ({elapsed:.1f}s)",
        f"Embedded {report.embedded} new texts, reused {report.reused} saved.",
        "",
        *_project_lines(report),
        "",
        *_skipped_lines(report),
        f"Index: {index_dir}",
    ]
    return "\n".join(lines)


def _format_duration(seconds: int) -> str:
    return "under a minute" if seconds < 60 else f"~{math.ceil(seconds / 60)} min"


def _existing_chunk_count(index_dir: Path) -> int:
    """Chunks already in the index - 0 when there is none. Never creates the directory,
    and treats an empty collection (e.g. left by a failed first run) as no index."""
    if not index_dir.exists():
        return 0
    return ChunkStore(index_dir).count()


def _index_unchanged_note(existing_chunks: int) -> str:
    if existing_chunks:
        return f"Nothing was written - the existing index ({existing_chunks} chunks) is unchanged."
    return "Nothing was written - no index has been built yet."


def _embedding_header(plan: EmbeddingPlan, total: int) -> str:
    if plan.to_embed == 0:
        return f"All {total} chunks already embedded - reusing saved embeddings, no API calls."
    if plan.reused:
        return (
            f"Embedding {plan.to_embed} texts - {plan.reused} of {total} chunks "
            "reuse saved embeddings."
        )
    return f"Embedding {plan.to_embed} texts from {total} chunks..."


def _format_dry_run(report: IndexReport, plan: EmbeddingPlan) -> str:
    requests = len(plan_batches(plan.pending))
    seconds = estimate_embedding_seconds(plan.pending)
    saved = (
        [f"Saved embeddings: {plan.reused} of {report.chunks} chunks already embedded."]
        if plan.reused
        else []
    )
    lines = [
        "Dry run - nothing embedded, nothing written.",
        "",
        f"Would index {report.documents} documents across {report.projects} projects "
        f"-> {report.chunks} chunks",
        "",
        *_project_lines(report),
        "",
        *_skipped_lines(report),
        *saved,
        f"Estimated embedding requests: {requests} "
        f"({plan.to_embed} texts, batch size {DEFAULT_BATCH_SIZE})",
        f"Estimated time: {_format_duration(seconds)} (free tier: one batch per minute)",
    ]
    return "\n".join(lines)


@click.group()
@click.version_option(package_name="second-brain-search")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Semantic search over your own project documentation."""
    # Tests inject `index_dir`, `cache_path` and `embedder_factory` here;
    # production leaves it empty.
    ctx.ensure_object(dict)


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Emit the raw document list as JSON.")
def discover(as_json: bool) -> None:
    """List the documentation files that would be indexed."""
    config = _load_config_or_fail()
    docs = discover_documents(config)

    if as_json:
        click.echo(json.dumps([_as_dict(d) for d in docs], indent=2))
        return

    if not docs:
        click.echo(f"No documents found under {config.scan_root}.")
        click.echo("Check include_patterns / exclude_paths in config.toml.")
        sys.exit(1)

    grouped: dict[str, list[DiscoveredDoc]] = defaultdict(list)
    for doc in docs:
        grouped[doc.project].append(doc)

    click.echo(f"Scanning {config.scan_root}\n")
    for project in sorted(grouped, key=str.lower):
        items = grouped[project]
        total = sum(d.size_bytes for d in items)
        click.echo(f"  {project}  ({len(items)} files, {_human_bytes(total)})")
        for doc in items:
            click.echo(f"      {doc.rel_path}")
        click.echo("")

    click.echo(
        f"{len(docs)} documents across {len(grouped)} projects, "
        f"{_human_bytes(sum(d.size_bytes for d in docs))} total."
    )


@cli.command()
@click.option(
    "--dry-run",
    is_flag=True,
    help="Discover and chunk, then report what would be embedded. No API key, no writes.",
)
@click.pass_context
def index(ctx: click.Context, dry_run: bool) -> None:
    """Rebuild the search index from scratch."""
    started = time.perf_counter()
    config = _load_config_or_fail()
    index_dir = Path(ctx.obj.get("index_dir", DEFAULT_INDEX_DIR))
    cache_path = Path(ctx.obj.get("cache_path", DEFAULT_CACHE_PATH))

    click.echo(f"Scanning {config.scan_root}")
    collected = collect_chunks(config)
    total = len(collected.chunks)

    if dry_run:
        if not collected.chunks:
            raise click.ClickException(
                f"Nothing to index under {config.scan_root} - no chunks were produced. "
                "Check SBS_SCAN_ROOT and config.toml."
            )
        # Read-only: a dry run must not create the cache. It builds no embedder
        # (no key needed), so it assumes the production Gemini vector space.
        plan = embedding_plan(
            collected.chunks,
            gemini_cache_namespace(EMBEDDING_MODEL, DEFAULT_DIMENSIONS),
            EmbeddingCache.open_existing(cache_path),
        )
        click.echo(_format_dry_run(collected.report, plan))
        return

    # Counted before anything opens the store for writing, so failure messages
    # describe the index as it was, not as a half-started run left it.
    existing_chunks = _existing_chunk_count(index_dir)

    # A mistyped SBS_SCAN_ROOT pointing at an empty folder would otherwise
    # rebuild a good index into an empty one.
    if not collected.chunks:
        raise click.ClickException(
            f"Nothing to index under {config.scan_root} - no chunks were produced. "
            f"{_index_unchanged_note(existing_chunks)} Check SBS_SCAN_ROOT and config.toml."
        )

    factory: EmbedderFactory = ctx.obj.get("embedder_factory", _gemini_embedder)
    embedder: Embedder | None = None
    cache: EmbeddingCache | None = None
    try:
        embedder = factory(config)
        cache = EmbeddingCache(cache_path)
        click.echo(
            _embedding_header(embedding_plan(collected.chunks, embedder.cache_namespace, cache), total)
        )
        report = store_chunks(
            collected,
            embedder,
            ChunkStore(index_dir),
            rebuild=True,
            on_batch=lambda n, batches: click.echo(f"Embedding batch {n}/{batches}..."),
            cache=cache,
        )
    except EmbeddingError as exc:
        notes = [str(exc)]
        if embedder is not None and cache is not None:
            after = embedding_plan(collected.chunks, embedder.cache_namespace, cache)
            if after.reused:
                notes.append(
                    f"{after.reused} of {total} chunks are embedded and saved - "
                    f"the next run embeds only the remaining {after.to_embed} texts."
                )
        notes.append(_index_unchanged_note(existing_chunks))
        raise click.ClickException(" ".join(notes)) from exc

    click.echo(
        format_index_report(report, elapsed=time.perf_counter() - started, index_dir=index_dir)
    )


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[:SNIPPET_CHARS].rstrip() + "..."


def _citation(result: RetrievedChunk) -> str:
    location = f"{result.project} / {result.rel_path}"
    return f"{location} > {result.heading_path}" if result.heading_path else location


def format_search_results(query: str, results: Sequence[RetrievedChunk]) -> str:
    lines = [f'Top {len(results)} for "{query}":', ""]
    for rank, result in enumerate(results, start=1):
        lines.append(f"{rank}. [{result.score:.2f}] {_citation(result)}")
        lines.append(f"   {_snippet(result.text)}")
        lines.append("")
    return "\n".join(lines).rstrip()


@cli.command()
@click.argument("query")
@click.option(
    "-k",
    "k",
    type=click.IntRange(1, 50),
    default=DEFAULT_K,
    show_default=True,
    help="How many chunks to return.",
)
@click.option("--project", default=None, help="Only search this project (case-insensitive).")
@click.pass_context
def search(ctx: click.Context, query: str, k: int, project: str | None) -> None:
    """Show the chunks most relevant to QUERY, with citations. Costs one embedded text."""
    config = _load_config_or_fail()
    index_dir = Path(ctx.obj.get("index_dir", DEFAULT_INDEX_DIR))

    # Checked before opening the store, which would otherwise create .chroma/.
    if _existing_chunk_count(index_dir) == 0:
        raise click.ClickException("There is no index yet - run `sbs index` first.")

    factory: EmbedderFactory = ctx.obj.get("embedder_factory", _gemini_embedder)
    try:
        results = retrieve(query, factory(config), ChunkStore(index_dir), k=k, project=project)
    except (RetrievalError, EmbeddingError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(format_search_results(query, results))


@cli.command("eval")
@click.option(
    "--fixture",
    "fixture",
    type=click.Path(path_type=Path),
    default=None,
    help=f"Eval suite TOML. Default: {DEFAULT_EVAL_SUITE}",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(path_type=Path),
    default=None,
    help=f"Where to write the JSON report. Default: {DEFAULT_EVAL_OUT_DIR}",
)
@click.option(
    "-k", "k", type=click.IntRange(1, 50), default=DEFAULT_ANSWER_K, show_default=True,
    help="Passages retrieved per question.",
)
@click.pass_context
def run_eval_command(ctx: click.Context, fixture: Path | None, out_dir: Path | None, k: int) -> None:
    """Run the retrieval eval suite and write a JSON report.

    Retrieval only - nothing is generated. Costs one embedded text per question.
    """
    config = _load_config_or_fail()
    index_dir = Path(ctx.obj.get("index_dir", DEFAULT_INDEX_DIR))
    suite_path = Path(fixture) if fixture is not None else DEFAULT_EVAL_SUITE
    reports_dir = Path(out_dir) if out_dir is not None else DEFAULT_EVAL_OUT_DIR

    # Checked before the store is opened, which would create .chroma/.
    if _existing_chunk_count(index_dir) == 0:
        raise click.ClickException("There is no index yet - run `sbs index` first.")

    try:
        suite = load_eval_suite(suite_path)
    except EvalSuiteError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"{suite.path.name}: {len(suite.questions)} questions")
    click.echo(f"This spends {len(suite.questions)} embedded texts of today's quota.\n")

    factory: EmbedderFactory = ctx.obj.get("embedder_factory", _gemini_embedder)
    try:
        report = run_eval(suite, factory(config), ChunkStore(index_dir), k=k)
    except (RetrievalError, EmbeddingError) as exc:
        raise click.ClickException(str(exc)) from exc

    for outcome in report.outcomes:
        if outcome.expects_no_answer:
            best = f"{outcome.top_score:.2f}" if outcome.top_score is not None else "none"
            verdict = f"expects no answer (top score {best})"
        elif outcome.missing:
            verdict = f"missing {', '.join(outcome.missing)}"
        else:
            verdict = f"found {', '.join(outcome.found)}"
        click.echo(f"  {outcome.question.id:28} {verdict}")

    summary = report.summary
    click.echo(
        f"\n{summary['questions']} questions, {summary['expecting_an_answer']} expecting an answer; "
        f"every expected project found in {summary['all_expected_found']}, "
        f"at least one in {summary['any_expected_found']}."
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.strftime("%Y%m%dT%H%M%SZ")
    destination = reports_dir / f"retrieval-v{suite.version}-{stamp}.json"
    destination.write_text(
        json.dumps(report_to_dict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    click.echo(f"Report: {destination}")


def _gemini_generator(config: Config) -> Generator:
    return GeminiGenerator(api_key=config.gemini_api_key, model=config.generation_model)


def _source_line(citation: Citation) -> str:
    dated = f"{citation.date_source} {citation.date}" if citation.date != "unknown" else "undated"
    if citation.changed_since_indexed:
        dated += ", changed since indexed"
    return f"  [{citation.number}] {_citation(citation.chunk)}  ({dated})"


def _passage_lines(passages: Sequence[RetrievedChunk], *, start: int = 1) -> list[str]:
    lines = []
    for rank, passage in enumerate(passages[start - 1 :], start=start):
        lines.append(f"  {rank}. [{passage.score:.2f}] {_citation(passage)}")
        lines.append(f"     {_snippet(passage.text)}")
    return lines


def format_answer(answer: Answer, *, show_context: bool = False) -> str:
    lines: list[str] = []

    if answer.answerable:
        lines += [answer.text, "", "Sources:"]
        lines += [_source_line(c) for c in answer.citations]
    else:
        lines += [f'No answer in your documentation for: "{answer.question}"', ""]
        if answer.passages:
            # Shown so a refusal can be checked rather than just believed - and
            # labelled, because these passages are what did NOT answer it.
            lines += ["Closest passages (not an answer):"]
            lines += _passage_lines(answer.passages[:REFUSAL_PASSAGES])

    if answer.warnings:
        lines += [""] + [f"Warning: {w}" for w in answer.warnings]

    if show_context:
        lines += ["", f"Context ({len(answer.passages)} passages retrieved):"]
        lines += _passage_lines(answer.passages)

    cited = len(answer.citations)
    lines += [
        "",
        f"{answer.model} - {len(answer.passages)} passages retrieved, {cited} cited",
    ]
    return "\n".join(lines)


@cli.command()
@click.argument("question")
@click.option(
    "-k",
    "k",
    type=click.IntRange(1, 50),
    default=DEFAULT_ANSWER_K,
    show_default=True,
    help="How many passages to give the model.",
)
@click.option("--project", default=None, help="Only use this project (case-insensitive).")
@click.option("--show-context", is_flag=True, help="Also print every passage retrieved.")
@click.pass_context
def ask(ctx: click.Context, question: str, k: int, project: str | None, show_context: bool) -> None:
    """Answer QUESTION from your documentation, with citations.

    Costs one embedded text and one generation request. An answer that cites
    nothing is shown as a refusal - there would be nothing to check it against.
    """
    config = _load_config_or_fail()
    index_dir = Path(ctx.obj.get("index_dir", DEFAULT_INDEX_DIR))

    # Checked before opening the store, which would otherwise create .chroma/.
    if _existing_chunk_count(index_dir) == 0:
        raise click.ClickException("There is no index yet - run `sbs index` first.")

    embedder_factory: EmbedderFactory = ctx.obj.get("embedder_factory", _gemini_embedder)
    generator_factory: GeneratorFactory = ctx.obj.get("generator_factory", _gemini_generator)

    try:
        answer = answer_question(
            question,
            embedder_factory(config),
            ChunkStore(index_dir),
            generator_factory(config),
            k=k,
            project=project,
        )
    except (RetrievalError, EmbeddingError, GenerationError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(format_answer(answer, show_context=show_context))


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point."""
    force_utf8_streams(sys.stdout, sys.stderr)
    cli.main(args=list(argv) if argv is not None else None, prog_name="sbs")
