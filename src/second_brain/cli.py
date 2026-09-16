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

from .config import Config, ConfigError, load_config
from .discovery import DiscoveredDoc, discover_documents
from .embedding import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIMENSIONS,
    EMBEDDING_MODEL,
    Embedder,
    EmbeddingError,
    GeminiEmbedder,
    estimate_embedding_seconds,
    gemini_cache_namespace,
)
from .embedding_cache import EmbeddingCache
from .pipeline import EmbeddingPlan, IndexReport, collect_chunks, embedding_plan, store_chunks
from .store import ChunkStore

# src/second_brain/cli.py -> repo root. Both gitignored.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = _REPO_ROOT / ".chroma"
DEFAULT_CACHE_PATH = _REPO_ROOT / "data" / "embedding_cache.sqlite"

EmbedderFactory = Callable[[Config], Embedder]


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
    requests = math.ceil(plan.to_embed / DEFAULT_BATCH_SIZE)
    seconds = estimate_embedding_seconds(plan.to_embed)
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


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point."""
    force_utf8_streams(sys.stdout, sys.stderr)
    cli.main(args=list(argv) if argv is not None else None, prog_name="sbs")
