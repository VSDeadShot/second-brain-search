"""End-to-end indexing, offline: discover -> read -> chunk -> embed -> store."""

from __future__ import annotations

from pathlib import Path

import pytest

from second_brain.config import Config, DEFAULT_EXCLUDE_DIRS, DEFAULT_INCLUDE_PATTERNS
from second_brain.discovery import discover_documents
from second_brain.embedding import EmbeddingError
from second_brain.pipeline import collect_chunks, index_documents, store_chunks
from second_brain.store import ChunkStore

from conftest import make_file, make_git_dir
from fakes import FailingEmbedder, FakeEmbedder


def config_for(root: Path) -> Config:
    return Config(
        scan_root=root,
        gemini_api_key=None,
        include_patterns=DEFAULT_INCLUDE_PATTERNS,
        include_dirs=("docs",),
        exclude_dirs=DEFAULT_EXCLUDE_DIRS,
        exclude_paths=(),
    )


def test_indexes_every_discovered_document(tmp_path: Path, corpus: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")

    report = index_documents(config_for(corpus), FakeEmbedder(), store)

    assert report.documents == 3
    assert report.projects == 2
    assert report.chunks > 0


def test_report_chunk_count_matches_the_store(tmp_path: Path, corpus: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")

    report = index_documents(config_for(corpus), FakeEmbedder(), store)

    assert store.count() == report.chunks


def test_reindexing_does_not_duplicate(tmp_path: Path, corpus: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")
    config = config_for(corpus)

    first = index_documents(config, FakeEmbedder(), store)
    second = index_documents(config, FakeEmbedder(), store)

    assert store.count() == first.chunks == second.chunks


def test_every_chunk_is_embedded_once(tmp_path: Path, corpus: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")
    embedder = FakeEmbedder()

    report = index_documents(config_for(corpus), embedder, store)

    embedded = sum(len(batch) for batch in embedder.embed_calls)
    assert embedded == report.chunks


def test_stored_chunks_carry_their_project(tmp_path: Path, corpus: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")

    index_documents(config_for(corpus), FakeEmbedder(), store)

    projects = {m["project"] for m in store.all_metadata()}
    assert projects == {"Alpha", "Beta"}


def test_empty_document_is_counted_but_yields_no_chunks(tmp_path: Path) -> None:
    root = tmp_path / "root"
    make_file(make_git_dir(root / "Solo") / "README.md", "   \n\n")
    store = ChunkStore(tmp_path / "chroma")

    report = index_documents(config_for(root), FakeEmbedder(), store)

    assert report.documents == 1
    assert report.chunks == 0


def test_undecodable_bytes_do_not_crash_the_run(tmp_path: Path) -> None:
    """Self-authored markdown should be UTF-8, but one bad file must not stop a rebuild."""
    root = tmp_path / "root"
    proj = make_git_dir(root / "Solo")
    (proj / "README.md").write_bytes(
        b"# Title\n\nvalid text \xff\xfe with invalid bytes, padded out far enough "
        b"that the chunk clears the size floor.\n"
    )
    store = ChunkStore(tmp_path / "chroma")

    report = index_documents(config_for(root), FakeEmbedder(), store)

    assert report.chunks > 0
    assert report.skipped == []


def test_file_deleted_after_discovery_is_skipped(tmp_path: Path, corpus: Path) -> None:
    """Discovery walks first, so a file can vanish before the pipeline reads it."""
    store = ChunkStore(tmp_path / "chroma")
    config = config_for(corpus)

    docs = discover_documents(config)
    (corpus / "Beta" / "README.md").unlink()

    report = index_documents(config, FakeEmbedder(), store, docs=docs)

    assert report.documents == 2
    assert len(report.skipped) == 1
    assert "Beta" in report.skipped[0][0]


def test_plain_upsert_leaves_stale_chunks_behind(tmp_path: Path, corpus: Path) -> None:
    """Why rebuild exists: without it, a deleted document stays searchable."""
    store = ChunkStore(tmp_path / "chroma")
    config = config_for(corpus)
    index_documents(config, FakeEmbedder(), store)

    (corpus / "Beta" / "README.md").unlink()
    index_documents(config, FakeEmbedder(), store)

    assert "Beta" in {m["project"] for m in store.all_metadata()}


def test_rebuild_removes_chunks_of_a_deleted_document(tmp_path: Path, corpus: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")
    config = config_for(corpus)
    index_documents(config, FakeEmbedder(), store, rebuild=True)

    (corpus / "Beta" / "README.md").unlink()
    report = index_documents(config, FakeEmbedder(), store, rebuild=True)

    assert {m["project"] for m in store.all_metadata()} == {"Alpha"}
    assert store.count() == report.chunks


def test_rebuild_removes_extra_chunks_of_a_shrunk_document(tmp_path: Path) -> None:
    """Deterministic ids mean chunks past the new end would otherwise survive."""
    root = tmp_path / "root"
    readme = make_git_dir(root / "Solo") / "README.md"
    section = "A section body long enough to become its own chunk in the index."
    make_file(readme, "\n\n".join(f"# Part {i}\n\n{section}" for i in range(5)))
    store = ChunkStore(tmp_path / "chroma")
    config = config_for(root)
    assert index_documents(config, FakeEmbedder(), store, rebuild=True).chunks == 5

    make_file(readme, f"# Part 0\n\n{section}")
    report = index_documents(config, FakeEmbedder(), store, rebuild=True)

    assert report.chunks == 1
    assert store.count() == 1


def test_failed_embed_during_rebuild_keeps_the_previous_index(
    tmp_path: Path, corpus: Path
) -> None:
    """Resetting before embedding would turn one API error into an empty index."""
    store = ChunkStore(tmp_path / "chroma")
    config = config_for(corpus)
    before = index_documents(config, FakeEmbedder(), store, rebuild=True)

    with pytest.raises(EmbeddingError):
        index_documents(config, FailingEmbedder(), store, rebuild=True)

    assert store.count() == before.chunks
    assert {m["project"] for m in store.all_metadata()} == {"Alpha", "Beta"}


def test_report_counts_chunks_per_project(tmp_path: Path, corpus: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")

    report = index_documents(config_for(corpus), FakeEmbedder(), store)

    assert set(report.chunks_per_project) == {"Alpha", "Beta"}
    assert sum(report.chunks_per_project.values()) == report.chunks


def test_store_chunks_forwards_batch_progress(tmp_path: Path, corpus: Path) -> None:
    seen: list[tuple[int, int]] = []

    store_chunks(
        collect_chunks(config_for(corpus)),
        FakeEmbedder(),
        ChunkStore(tmp_path / "chroma"),
        on_batch=lambda n, total: seen.append((n, total)),
    )

    assert seen == [(1, 1)]


def test_collect_chunks_matches_what_indexing_stores(tmp_path: Path, corpus: Path) -> None:
    """The dry run relies on this: same chunks, no embedder, no store."""
    config = config_for(corpus)
    store = ChunkStore(tmp_path / "chroma")

    collected = collect_chunks(config)
    report = index_documents(config, FakeEmbedder(), store)

    assert len(collected.chunks) == report.chunks
    assert collected.report.chunks_per_project == report.chunks_per_project
    assert {c.chunk_id for c in collected.chunks} == {m for m in store.all_ids()}
