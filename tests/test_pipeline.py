"""End-to-end indexing, offline: discover -> read -> chunk -> embed -> store."""

from __future__ import annotations

from pathlib import Path

import pytest

from second_brain.config import Config, DEFAULT_EXCLUDE_DIRS, DEFAULT_INCLUDE_PATTERNS
from second_brain.discovery import discover_documents
from second_brain.pipeline import index_documents
from second_brain.store import ChunkStore

from conftest import make_file, make_git_dir
from fakes import FakeEmbedder


def config_for(root: Path) -> Config:
    return Config(
        scan_root=root,
        gemini_api_key=None,
        include_patterns=DEFAULT_INCLUDE_PATTERNS,
        include_dirs=("docs",),
        exclude_dirs=DEFAULT_EXCLUDE_DIRS,
        exclude_paths=(),
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    alpha = make_git_dir(root / "Alpha")
    make_file(
        alpha / "README.md",
        "# Alpha\n\nAlpha does a thing worth describing at realistic length.\n\n"
        "## Detail\n\nMore detail about how the thing actually works in practice.",
    )
    make_file(
        alpha / "CLAUDE.md",
        "# Notes\n\nSome notes about Alpha, long enough to clear the size floor.",
    )
    beta = make_git_dir(root / "Beta")
    make_file(
        beta / "README.md",
        "# Beta\n\nBeta is different from Alpha, and says so at some length.",
    )
    return root


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
