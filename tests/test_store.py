"""Chroma persistence. Uses a real store in tmp_path - no mocking of the database."""

from __future__ import annotations

from pathlib import Path

import pytest

from second_brain.chunking import chunk_document
from second_brain.store import ChunkStore

from fakes import FakeEmbedder


REALISTIC = (
    "# Title\n\nThis body is long enough to clear the chunking size floor, "
    "the way any real project document would be."
)


def build(doc_factory, text: str = REALISTIC):
    chunks = chunk_document(doc_factory(), text)
    vectors = FakeEmbedder().embed_documents([c.text for c in chunks])
    return chunks, vectors


def test_upsert_then_count(tmp_path: Path, doc_factory) -> None:
    store = ChunkStore(tmp_path / "chroma")
    chunks, vectors = build(doc_factory)

    store.upsert(chunks, vectors)

    assert store.count() == len(chunks)


def test_upsert_is_idempotent(tmp_path: Path, doc_factory) -> None:
    """Re-indexing the same document must overwrite, never duplicate."""
    store = ChunkStore(tmp_path / "chroma")
    chunks, vectors = build(doc_factory)

    store.upsert(chunks, vectors)
    store.upsert(chunks, vectors)

    assert store.count() == len(chunks)


def test_citation_metadata_round_trips(tmp_path: Path, doc_factory) -> None:
    store = ChunkStore(tmp_path / "chroma")
    chunks, vectors = build(
        doc_factory,
        "# Architecture\n\n## Caching\n\nThe caching detail, written out at the "
        "sort of length a real document would actually have.",
    )
    store.upsert(chunks, vectors)

    stored = store.get(chunks[-1].chunk_id)

    assert stored["project"] == chunks[-1].project
    assert stored["rel_path"] == chunks[-1].rel_path
    assert stored["heading_path"] == "Architecture > Caching"
    assert stored["chunk_index"] == chunks[-1].chunk_index
    assert stored["content_hash"] == chunks[-1].content_hash


def test_document_text_is_retrievable(tmp_path: Path, doc_factory) -> None:
    """Slice 5 cites the text, so it has to come back out, not just the vector."""
    store = ChunkStore(tmp_path / "chroma")
    chunks, vectors = build(doc_factory)
    store.upsert(chunks, vectors)

    assert store.get(chunks[0].chunk_id)["text"] == chunks[0].text


def test_data_survives_a_new_store_instance(tmp_path: Path, doc_factory) -> None:
    path = tmp_path / "chroma"
    chunks, vectors = build(doc_factory)
    ChunkStore(path).upsert(chunks, vectors)

    assert ChunkStore(path).count() == len(chunks)


def test_reset_empties_the_collection(tmp_path: Path, doc_factory) -> None:
    store = ChunkStore(tmp_path / "chroma")
    chunks, vectors = build(doc_factory)
    store.upsert(chunks, vectors)

    store.reset()

    assert store.count() == 0


def test_mismatched_vector_count_is_rejected(tmp_path: Path, doc_factory) -> None:
    store = ChunkStore(tmp_path / "chroma")
    chunks, vectors = build(doc_factory)

    with pytest.raises(ValueError, match="counts must match"):
        store.upsert(chunks, vectors[:-1] if len(vectors) > 1 else [])


def test_upserting_nothing_is_harmless(tmp_path: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")

    store.upsert([], [])

    assert store.count() == 0
