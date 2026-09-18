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


# --- vector space recorded in the index -----------------------------------------


def test_a_new_store_has_no_recorded_vector_space(tmp_path: Path) -> None:
    assert ChunkStore(tmp_path / "chroma").namespace is None


def test_reset_can_record_the_vector_space(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    ChunkStore(path).reset(namespace="model|768|RETRIEVAL_DOCUMENT")

    assert ChunkStore(path).namespace == "model|768|RETRIEVAL_DOCUMENT"


def test_recording_a_space_on_an_existing_store_persists(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    ChunkStore(path).record_namespace("keyword|256")

    assert ChunkStore(path).namespace == "keyword|256"


def _cosine_distance_check(store: ChunkStore, doc_factory) -> float:
    """Store [3, 4] and query [0.6, 0.8]: cosine distance 0, L2 distance ~16."""
    chunk = chunk_document(doc_factory(), REALISTIC)[0]
    store.upsert([chunk], [[3.0, 4.0]])
    return store.query([0.6, 0.8], k=1)[0]["distance"]


def test_reset_with_a_namespace_keeps_cosine_distance(tmp_path: Path, doc_factory) -> None:
    store = ChunkStore(tmp_path / "chroma")
    store.reset(namespace="ns")

    assert _cosine_distance_check(store, doc_factory) == pytest.approx(0.0, abs=1e-6)


def test_recording_a_namespace_keeps_cosine_distance(tmp_path: Path, doc_factory) -> None:
    """Probed on chromadb 1.5.9: recording the namespace drops 'hnsw:space' from the
    visible metadata, yet the collection keeps cosine. Pinned so an upgrade can't
    silently switch retrieval to L2."""
    store = ChunkStore(tmp_path / "chroma")
    store.record_namespace("ns")

    assert _cosine_distance_check(ChunkStore(tmp_path / "chroma"), doc_factory) == pytest.approx(
        0.0, abs=1e-6
    )


# --- query ----------------------------------------------------------------------


def two_project_store(tmp_path: Path, doc_factory) -> tuple[ChunkStore, list]:
    store = ChunkStore(tmp_path / "chroma")
    alpha = chunk_document(doc_factory("README.md", "Alpha"), REALISTIC)[0]
    beta = chunk_document(doc_factory("README.md", "Beta"), REALISTIC)[0]
    store.upsert([alpha, beta], [[1.0, 0.0], [0.0, 1.0]])
    return store, [alpha, beta]


def test_query_returns_nearest_first_with_text_metadata_and_distance(
    tmp_path: Path, doc_factory
) -> None:
    store, (alpha, beta) = two_project_store(tmp_path, doc_factory)

    rows = store.query([0.9, 0.1], k=2)

    assert [r["id"] for r in rows] == [alpha.chunk_id, beta.chunk_id]
    assert rows[0]["distance"] < rows[1]["distance"]
    assert rows[0]["text"] == alpha.text
    assert rows[0]["project"] == "Alpha"
    assert rows[0]["content_hash"] == alpha.content_hash


def test_query_can_be_limited_to_one_project(tmp_path: Path, doc_factory) -> None:
    store, (alpha, beta) = two_project_store(tmp_path, doc_factory)

    rows = store.query([1.0, 0.0], k=5, project="Beta")

    assert [r["id"] for r in rows] == [beta.chunk_id]


def test_query_on_an_empty_store_returns_nothing(tmp_path: Path) -> None:
    assert ChunkStore(tmp_path / "chroma").query([1.0, 0.0], k=5) == []


def test_projects_lists_every_indexed_project(tmp_path: Path, doc_factory) -> None:
    store, _ = two_project_store(tmp_path, doc_factory)

    assert store.projects() == ["Alpha", "Beta"]
