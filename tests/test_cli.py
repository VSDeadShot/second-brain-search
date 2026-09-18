"""CLI: `sbs index`, `--dry-run`, and console encoding.

Every invocation injects a tmp index dir, a tmp embedding cache and, where
embedding happens, a fake embedder through click's context object - no test
touches the real .chroma/ or data/, or calls the Gemini API.
"""

from __future__ import annotations

import io
import sys
import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from second_brain.cli import (
    DEFAULT_CACHE_PATH,
    cli,
    force_utf8_streams,
    format_index_report,
    main,
)
from second_brain.config import Config, DEFAULT_EXCLUDE_DIRS, DEFAULT_INCLUDE_PATTERNS
from second_brain.embedding import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIMENSIONS,
    EMBEDDING_MODEL,
    GeminiEmbedder,
    gemini_cache_namespace,
)
from second_brain.embedding_cache import EmbeddingCache, cache_key
from second_brain.pipeline import IndexReport, collect_chunks
from second_brain.store import ChunkStore

from conftest import make_file, make_git_dir
from second_brain.embedding import DailyQuotaExceeded

from fakes import (
    FailingEmbedder,
    FakeClock,
    FakeEmbedder,
    KeywordEmbedder,
    StubClient,
    StubModels,
    daily_quota_error,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def cache_path_for(index_dir: Path) -> Path:
    return index_dir.parent / "embedding_cache.sqlite"


def run(args: list[str], *, scan_root: Path, index_dir: Path, factory=None, api_key="fake-key"):
    obj: dict = {"index_dir": index_dir, "cache_path": cache_path_for(index_dir)}
    if factory is not None:
        obj["embedder_factory"] = factory
    return CliRunner().invoke(
        cli,
        args,
        obj=obj,
        env={"SBS_SCAN_ROOT": str(scan_root), "GEMINI_API_KEY": api_key},
    )


def fake_factory(config):
    return FakeEmbedder()


def must_not_embed(config):
    raise AssertionError("embedder must not be built")


# --- sbs index -----------------------------------------------------------------


def test_index_populates_the_store_and_reports(tmp_path: Path, corpus: Path) -> None:
    index_dir = tmp_path / "chroma"

    result = run(["index"], scan_root=corpus, index_dir=index_dir, factory=fake_factory)

    assert result.exit_code == 0, result.output
    assert "Indexed 3 documents across 2 projects" in result.output
    store = ChunkStore(index_dir)
    assert f"-> {store.count()} chunks" in result.output


def test_index_lists_chunk_counts_per_project(tmp_path: Path, corpus: Path) -> None:
    result = run(["index"], scan_root=corpus, index_dir=tmp_path / "chroma", factory=fake_factory)

    assert "Alpha" in result.output
    assert "Beta" in result.output


def test_index_is_a_full_rebuild(tmp_path: Path, corpus: Path) -> None:
    """SPEC decision 5: a deleted document must not survive the next index."""
    index_dir = tmp_path / "chroma"
    run(["index"], scan_root=corpus, index_dir=index_dir, factory=fake_factory)

    (corpus / "Beta" / "README.md").unlink()
    result = run(["index"], scan_root=corpus, index_dir=index_dir, factory=fake_factory)

    assert result.exit_code == 0, result.output
    assert {m["project"] for m in ChunkStore(index_dir).all_metadata()} == {"Alpha"}


def test_missing_api_key_is_a_clean_error_and_keeps_the_index(
    tmp_path: Path, corpus: Path
) -> None:
    index_dir = tmp_path / "chroma"
    run(["index"], scan_root=corpus, index_dir=index_dir, factory=fake_factory)
    before = ChunkStore(index_dir).count()

    # No factory override: the real GeminiEmbedder is built, and must refuse.
    result = run(["index"], scan_root=corpus, index_dir=index_dir, api_key="")

    assert result.exit_code != 0
    assert "GEMINI_API_KEY" in result.output
    assert "Traceback" not in result.output
    assert ChunkStore(index_dir).count() == before


def test_embedding_failure_is_a_clean_error_and_keeps_the_index(
    tmp_path: Path, corpus: Path
) -> None:
    index_dir = tmp_path / "chroma"
    run(["index"], scan_root=corpus, index_dir=index_dir, factory=fake_factory)
    before = ChunkStore(index_dir).count()

    result = run(
        ["index"], scan_root=corpus, index_dir=index_dir, factory=lambda c: FailingEmbedder()
    )

    assert result.exit_code != 0
    assert "simulated upstream failure" in result.output
    assert "Traceback" not in result.output
    assert ChunkStore(index_dir).count() == before


def test_empty_corpus_refuses_to_wipe_an_existing_index(tmp_path: Path, corpus: Path) -> None:
    """A mistyped SBS_SCAN_ROOT pointing at an empty folder must not erase a good index."""
    index_dir = tmp_path / "chroma"
    run(["index"], scan_root=corpus, index_dir=index_dir, factory=fake_factory)
    before = ChunkStore(index_dir).count()
    empty = tmp_path / "empty"
    empty.mkdir()

    result = run(["index"], scan_root=empty, index_dir=index_dir, factory=fake_factory)

    assert result.exit_code != 0
    assert "Nothing to index" in result.output
    assert f"the existing index ({before} chunks) is unchanged" in result.output
    assert ChunkStore(index_dir).count() == before


def test_config_error_is_clean(tmp_path: Path) -> None:
    result = run(
        ["index"],
        scan_root=tmp_path / "does-not-exist",
        index_dir=tmp_path / "chroma",
        factory=fake_factory,
    )

    assert result.exit_code != 0
    assert "SBS_SCAN_ROOT" in result.output
    assert "Traceback" not in result.output


# --- sbs index --dry-run -------------------------------------------------------


def test_dry_run_needs_no_key_and_writes_nothing(tmp_path: Path, corpus: Path) -> None:
    index_dir = tmp_path / "chroma"

    result = run(
        ["index", "--dry-run"],
        scan_root=corpus,
        index_dir=index_dir,
        factory=must_not_embed,
        api_key="",
    )

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert "Would index 3 documents across 2 projects" in result.output
    assert not index_dir.exists()


def test_dry_run_estimates_embedding_requests(tmp_path: Path) -> None:
    root = tmp_path / "root"
    body = "A section body long enough to become its own chunk in the index."
    chunk_count = DEFAULT_BATCH_SIZE + 1
    make_file(
        root / "Solo" / "README.md",
        "\n\n".join(f"# Part {i}\n\n{body}" for i in range(chunk_count)),
    )

    result = run(
        ["index", "--dry-run"],
        scan_root=root,
        index_dir=tmp_path / "chroma",
        factory=must_not_embed,
    )

    assert f"-> {chunk_count} chunks" in result.output
    assert "Estimated embedding requests: 2" in result.output
    assert "Estimated time: ~1 min" in result.output


def test_dry_run_single_batch_needs_under_a_minute(tmp_path: Path, corpus: Path) -> None:
    result = run(
        ["index", "--dry-run"],
        scan_root=corpus,
        index_dir=tmp_path / "chroma",
        factory=must_not_embed,
    )

    assert "Estimated time: under a minute" in result.output


# --- progress and pacing through the CLI ---------------------------------------


def stub_gemini_factory(models: StubModels):
    """Real GeminiEmbedder over a stubbed client: 2 texts per batch, instant fake clock."""
    clock = FakeClock()
    models.clock = clock

    def factory(config):
        return GeminiEmbedder(
            client=StubClient(models=models),
            dimensions=models.dimensions,
            batch_size=2,
            items_per_minute=2,
            sleep=clock.sleep,
            clock=clock.monotonic,
        )

    return factory


def test_index_shows_progress_for_each_batch(tmp_path: Path, corpus: Path) -> None:
    """4 chunks at 2 per minute."""
    models = StubModels()

    result = run(
        ["index"], scan_root=corpus, index_dir=tmp_path / "chroma", factory=stub_gemini_factory(models)
    )

    assert result.exit_code == 0, result.output
    assert "Embedding batch 1/2..." in result.output
    assert "Embedding batch 2/2..." in result.output
    assert models.call_times == [0.0, 60.0]


# --- daily quota and resuming ---------------------------------------------------


def test_default_cache_lives_in_the_gitignored_data_directory() -> None:
    assert DEFAULT_CACHE_PATH.parent == REPO_ROOT / "data"
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/" in ignored


def test_daily_quota_is_reported_clearly_without_retrying(tmp_path: Path, corpus: Path) -> None:
    models = StubModels(raise_queue=[daily_quota_error()])

    result = run(
        ["index"], scan_root=corpus, index_dir=tmp_path / "chroma", factory=stub_gemini_factory(models)
    )

    assert result.exit_code != 0
    assert "daily limit of 1000 embedded texts" in result.output
    assert "midnight Pacific" in result.output
    assert "Traceback" not in result.output
    assert len(models.calls) == 1


def test_failed_run_reports_the_progress_it_saved(tmp_path: Path, corpus: Path) -> None:
    models = StubModels(raise_on_calls={2: daily_quota_error()})

    result = run(
        ["index"], scan_root=corpus, index_dir=tmp_path / "chroma", factory=stub_gemini_factory(models)
    )

    assert result.exit_code != 0
    assert "2 of 4 chunks are embedded and saved" in result.output
    assert "the next run embeds only the remaining 2 texts" in result.output
    assert "no index has been built yet" in result.output


def test_resumed_index_embeds_only_what_is_missing(tmp_path: Path, corpus: Path) -> None:
    index_dir = tmp_path / "chroma"
    run(
        ["index"],
        scan_root=corpus,
        index_dir=index_dir,
        factory=stub_gemini_factory(StubModels(raise_on_calls={2: daily_quota_error()})),
    )

    resumed = StubModels()
    result = run(["index"], scan_root=corpus, index_dir=index_dir, factory=stub_gemini_factory(resumed))

    assert result.exit_code == 0, result.output
    assert "Embedding 2 texts - 2 of 4 chunks reuse saved embeddings" in result.output
    assert "Embedded 2 new texts, reused 2 saved." in result.output
    assert sum(len(call["contents"]) for call in resumed.calls) == 2
    assert ChunkStore(index_dir).count() == 4


def test_fully_saved_run_calls_the_api_not_at_all(tmp_path: Path, corpus: Path) -> None:
    index_dir = tmp_path / "chroma"
    run(["index"], scan_root=corpus, index_dir=index_dir, factory=stub_gemini_factory(StubModels()))

    again = StubModels()
    result = run(["index"], scan_root=corpus, index_dir=index_dir, factory=stub_gemini_factory(again))

    assert result.exit_code == 0, result.output
    assert "All 4 chunks already embedded" in result.output
    assert again.calls == []
    assert ChunkStore(index_dir).count() == 4


def test_dry_run_counts_saved_embeddings(tmp_path: Path, corpus: Path) -> None:
    """Tomorrow's dry run should say how much is left, not the full 812 again."""
    index_dir = tmp_path / "chroma"
    chunks = collect_chunks(
        Config(
            scan_root=corpus,
            gemini_api_key=None,
            include_patterns=DEFAULT_INCLUDE_PATTERNS,
            include_dirs=("docs",),
            exclude_dirs=DEFAULT_EXCLUDE_DIRS,
            exclude_paths=(),
        )
    ).chunks
    namespace = gemini_cache_namespace(EMBEDDING_MODEL, DEFAULT_DIMENSIONS)
    EmbeddingCache(cache_path_for(index_dir)).put_many(
        (cache_key(namespace, c.content_hash), [1.0]) for c in chunks[:2]
    )

    result = run(
        ["index", "--dry-run"], scan_root=corpus, index_dir=index_dir, factory=must_not_embed
    )

    assert result.exit_code == 0, result.output
    assert "Saved embeddings: 2 of 4 chunks already embedded" in result.output
    assert "Estimated embedding requests: 1 (2 texts, batch size 100)" in result.output


def test_dry_run_never_creates_the_cache(tmp_path: Path, corpus: Path) -> None:
    index_dir = tmp_path / "chroma"

    run(["index", "--dry-run"], scan_root=corpus, index_dir=index_dir, factory=must_not_embed)

    assert not cache_path_for(index_dir).exists()


# --- sbs search -----------------------------------------------------------------


def keyword_factory(config):
    return KeywordEmbedder()


def indexed_knowledge(tmp_path: Path, knowledge: Path) -> Path:
    index_dir = tmp_path / "chroma"
    result = run(["index"], scan_root=knowledge, index_dir=index_dir, factory=keyword_factory)
    assert result.exit_code == 0, result.output
    return index_dir


def search(args: list[str], *, knowledge: Path, index_dir: Path, factory=keyword_factory):
    return run(["search", *args], scan_root=knowledge, index_dir=index_dir, factory=factory)


def test_search_shows_ranked_results_with_citations(tmp_path: Path, knowledge: Path) -> None:
    index_dir = indexed_knowledge(tmp_path, knowledge)

    result = search(["how did I handle photo compression"], knowledge=knowledge, index_dir=index_dir)

    assert result.exit_code == 0, result.output
    first = next(line for line in result.output.splitlines() if line.startswith("1."))
    assert "Macro Tracker / CLAUDE.md > Photos" in first
    assert "[0." in first or "[1." in first


def test_search_respects_k(tmp_path: Path, knowledge: Path) -> None:
    index_dir = indexed_knowledge(tmp_path, knowledge)

    result = search(["caching", "-k", "1"], knowledge=knowledge, index_dir=index_dir)

    lines = result.output.splitlines()
    assert any(line.startswith("1.") for line in lines)
    assert not any(line.startswith("2.") for line in lines)


def test_search_can_be_limited_to_a_project(tmp_path: Path, knowledge: Path) -> None:
    index_dir = indexed_knowledge(tmp_path, knowledge)

    result = search(["photo compression", "--project", "rdbms"], knowledge=knowledge, index_dir=index_dir)

    ranked = [line for line in result.output.splitlines() if line[:1].isdigit()]
    assert ranked and all("RDBMS /" in line for line in ranked)


def test_search_rejects_k_outside_1_to_50(tmp_path: Path, knowledge: Path) -> None:
    index_dir = indexed_knowledge(tmp_path, knowledge)

    result = search(["caching", "-k", "0"], knowledge=knowledge, index_dir=index_dir)

    assert result.exit_code == 2  # click usage error


def test_search_snippets_are_one_line_and_truncated(tmp_path: Path) -> None:
    root = tmp_path / "root"
    long_body = "\n\n".join(f"caching paragraph number {i} with filler text here" for i in range(20))
    make_file(make_git_dir(root / "Solo") / "README.md", f"# Caching\n\n{long_body}")
    index_dir = tmp_path / "chroma"
    run(["index"], scan_root=root, index_dir=index_dir, factory=keyword_factory)

    result = search(["caching"], knowledge=root, index_dir=index_dir)

    lines = result.output.splitlines()
    snippet = lines[lines.index(next(l for l in lines if l.startswith("1."))) + 1]
    assert snippet.startswith("   ")
    assert snippet.rstrip().endswith("...")
    assert len(snippet) <= 3 + 240 + 3


def test_search_prints_emoji_from_the_corpus(tmp_path: Path) -> None:
    """11 of the real documents contain emoji; they must reach the terminal intact."""
    root = tmp_path / "root"
    make_file(
        make_git_dir(root / "Solo") / "README.md",
        "# Launch\n\nThe launch checklist is complete \U0001f680 and every caching step passed.",
    )
    index_dir = tmp_path / "chroma"
    run(["index"], scan_root=root, index_dir=index_dir, factory=keyword_factory)

    result = search(["launch checklist"], knowledge=root, index_dir=index_dir)

    assert "\U0001f680" in result.output


def test_search_without_an_index_is_clean_and_creates_nothing(tmp_path: Path, knowledge: Path) -> None:
    index_dir = tmp_path / "chroma"

    result = search(["caching"], knowledge=knowledge, index_dir=index_dir, factory=must_not_embed)

    assert result.exit_code != 0
    assert "sbs index" in result.output
    assert "Traceback" not in result.output
    assert not index_dir.exists()


def test_search_against_an_index_from_another_embedder_is_clean(
    tmp_path: Path, knowledge: Path
) -> None:
    index_dir = indexed_knowledge(tmp_path, knowledge)

    result = search(["caching"], knowledge=knowledge, index_dir=index_dir, factory=fake_factory)

    assert result.exit_code != 0
    assert "keyword|256" in result.output
    assert "Traceback" not in result.output


def test_search_unknown_project_is_clean(tmp_path: Path, knowledge: Path) -> None:
    index_dir = indexed_knowledge(tmp_path, knowledge)

    result = search(["caching", "--project", "Nope"], knowledge=knowledge, index_dir=index_dir)

    assert result.exit_code != 0
    assert "Watch Tracker" in result.output
    assert "Traceback" not in result.output


def test_search_daily_quota_is_clean(tmp_path: Path, knowledge: Path) -> None:
    index_dir = indexed_knowledge(tmp_path, knowledge)

    class QuotaSpent(KeywordEmbedder):
        def embed_query(self, text: str) -> list[float]:
            raise DailyQuotaExceeded("Gemini's free-tier daily limit of 1000 embedded texts is used up.")

    result = search(["caching"], knowledge=knowledge, index_dir=index_dir, factory=lambda c: QuotaSpent())

    assert result.exit_code != 0
    assert "daily limit" in result.output
    assert "Traceback" not in result.output


# --- failure wording ------------------------------------------------------------


def test_failure_with_no_index_does_not_claim_one_existed(tmp_path: Path, corpus: Path) -> None:
    result = run(
        ["index"],
        scan_root=corpus,
        index_dir=tmp_path / "chroma",
        factory=lambda c: FailingEmbedder(),
    )

    assert result.exit_code != 0
    assert "no index has been built yet" in result.output
    assert "existing index" not in result.output


def test_failure_with_an_index_reports_it_unchanged(tmp_path: Path, corpus: Path) -> None:
    index_dir = tmp_path / "chroma"
    run(["index"], scan_root=corpus, index_dir=index_dir, factory=fake_factory)
    chunks = ChunkStore(index_dir).count()

    result = run(["index"], scan_root=corpus, index_dir=index_dir, factory=lambda c: FailingEmbedder())

    assert f"the existing index ({chunks} chunks) is unchanged" in result.output


def test_empty_index_directory_is_not_treated_as_an_index(tmp_path: Path, corpus: Path) -> None:
    """The failed live run left an empty .chroma/ behind - that is not an index."""
    index_dir = tmp_path / "chroma"
    ChunkStore(index_dir)  # creates the directory and an empty collection

    result = run(["index"], scan_root=corpus, index_dir=index_dir, factory=lambda c: FailingEmbedder())

    assert "no index has been built yet" in result.output


# --- report formatting ---------------------------------------------------------


def test_report_lists_skipped_files_with_reasons() -> None:
    report = IndexReport(
        documents=1,
        chunks=2,
        projects=1,
        skipped=[("C:/x/README.md", "permission denied")],
        chunks_per_project={"X": 2},
    )

    text = format_index_report(report, elapsed=1.0, index_dir=Path("idx"))

    assert "Skipped 1 file" in text
    assert "C:/x/README.md" in text
    assert "permission denied" in text


# --- console encoding ----------------------------------------------------------


def cp1252_stream() -> io.TextIOWrapper:
    """What Windows hands Python when stdout is piped or redirected."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252")


def test_cp1252_stream_reproduces_the_crash() -> None:
    """Guards the other tests: without this, they could pass on a stream that never failed."""
    with pytest.raises(UnicodeEncodeError):
        cp1252_stream().write("launch \U0001f680")


def test_forcing_utf8_lets_emoji_through() -> None:
    stream = cp1252_stream()

    force_utf8_streams(stream)
    stream.write("launch \U0001f680")
    stream.flush()

    assert stream.buffer.getvalue().decode("utf-8") == "launch \U0001f680"


def test_streams_without_reconfigure_are_left_alone() -> None:
    stream = io.StringIO()

    force_utf8_streams(stream)  # must not raise

    stream.write("launch \U0001f680")
    assert stream.getvalue() == "launch \U0001f680"


def test_main_forces_utf8_before_running_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    out, err = cp1252_stream(), cp1252_stream()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    assert out.encoding == "utf-8"
    assert err.encoding == "utf-8"


def test_installed_sbs_script_points_at_main() -> None:
    """If the entry point still targets `cli`, the encoding fix is dead code."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["sbs"] == "second_brain.cli:main"
