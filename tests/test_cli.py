"""CLI: `sbs index`, `--dry-run`, and console encoding.

Every invocation injects a tmp index dir and, where embedding happens, a fake
embedder through click's context object - no test touches the real .chroma/
or calls the Gemini API.
"""

from __future__ import annotations

import io
import sys
import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from second_brain.cli import cli, force_utf8_streams, format_index_report, main
from second_brain.embedding import DEFAULT_BATCH_SIZE, GeminiEmbedder
from second_brain.pipeline import IndexReport
from second_brain.store import ChunkStore

from conftest import make_file
from fakes import FailingEmbedder, FakeClock, FakeEmbedder, StubClient, StubModels

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str], *, scan_root: Path, index_dir: Path, factory=None, api_key="fake-key"):
    obj: dict = {"index_dir": index_dir}
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


def test_index_shows_progress_for_each_batch(tmp_path: Path, corpus: Path) -> None:
    """Real GeminiEmbedder, stubbed client, fake clock: 4 chunks at 2 per minute."""
    clock = FakeClock()
    models = StubModels(clock=clock)

    def paced_factory(config):
        return GeminiEmbedder(
            client=StubClient(models=models),
            dimensions=models.dimensions,
            batch_size=2,
            items_per_minute=2,
            sleep=clock.sleep,
            clock=clock.monotonic,
        )

    result = run(["index"], scan_root=corpus, index_dir=tmp_path / "chroma", factory=paced_factory)

    assert result.exit_code == 0, result.output
    assert "Embedding batch 1/2..." in result.output
    assert "Embedding batch 2/2..." in result.output
    assert models.call_times == [0.0, 60.0]


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
