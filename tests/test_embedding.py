"""Embedding: batching, retry, and the shape of what comes back."""

from __future__ import annotations

import os

import pytest

from second_brain.embedding import EmbeddingError, GeminiEmbedder

from fakes import FakeEmbedder, StubClient, StubModels


def make_embedder(models: StubModels, **kwargs) -> GeminiEmbedder:
    defaults = dict(api_key="test-key", dimensions=models.dimensions, sleep=lambda _: None)
    defaults.update(kwargs)
    return GeminiEmbedder(client=StubClient(models=models), **defaults)


def test_returns_one_vector_per_input() -> None:
    models = StubModels()
    embedder = make_embedder(models)

    vectors = embedder.embed_documents(["a", "b", "c"])

    assert len(vectors) == 3
    assert all(len(v) == models.dimensions for v in vectors)


def test_empty_input_makes_no_api_call() -> None:
    models = StubModels()

    assert make_embedder(models).embed_documents([]) == []
    assert models.calls == []


def test_requests_are_batched() -> None:
    models = StubModels()
    embedder = make_embedder(models, batch_size=2)

    embedder.embed_documents(["a", "b", "c", "d", "e"])

    assert [len(c["contents"]) for c in models.calls] == [2, 2, 1]


def test_batching_preserves_order() -> None:
    """Vectors must line up with their inputs, or every citation is wrong."""
    models = StubModels()
    embedder = make_embedder(models, batch_size=2)

    embedder.embed_documents(["a", "b", "c"])

    sent = [t for call in models.calls for t in call["contents"]]
    assert sent == ["a", "b", "c"]


def test_transient_failures_are_retried() -> None:
    models = StubModels(failures_before_success=2)
    embedder = make_embedder(models, max_retries=3)

    vectors = embedder.embed_documents(["a"])

    assert len(vectors) == 1
    assert len(models.calls) == 3


def test_gives_up_after_max_retries() -> None:
    models = StubModels(failures_before_success=99)
    embedder = make_embedder(models, max_retries=2)

    with pytest.raises(EmbeddingError, match="after 2 attempts"):
        embedder.embed_documents(["a"])


def test_documents_and_queries_use_different_task_types() -> None:
    """Retrieval quality depends on asymmetric task types."""
    models = StubModels()
    embedder = make_embedder(models)

    embedder.embed_documents(["a"])
    embedder.embed_query("a")

    assert models.calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert models.calls[1]["config"].task_type == "RETRIEVAL_QUERY"


def test_requested_dimensionality_is_passed_through() -> None:
    models = StubModels(dimensions=4)
    embedder = make_embedder(models, dimensions=4)

    embedder.embed_documents(["a"])

    assert models.calls[0]["config"].output_dimensionality == 4


def test_unexpected_vector_count_is_an_error() -> None:
    """A silent length mismatch would misalign every chunk against its vector."""
    models = StubModels()
    embedder = make_embedder(models, batch_size=10)
    models.embed_content = lambda **kw: __import__("fakes").StubResponse(embeddings=[])

    with pytest.raises(EmbeddingError, match="expected 2"):
        embedder.embed_documents(["a", "b"])


def test_fake_embedder_is_deterministic() -> None:
    fake = FakeEmbedder()
    assert fake.embed_documents(["x"])[0] == fake.embed_documents(["x"])[0]
    assert fake.embed_query("x") == fake.embed_documents(["x"])[0]


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="live Gemini call; set GEMINI_API_KEY to run",
)
def test_live_gemini_embedding_roundtrip() -> None:
    """Opt-in: the only test that proves the real client and model name work."""
    embedder = GeminiEmbedder(api_key=os.environ["GEMINI_API_KEY"])

    vectors = embedder.embed_documents(["second brain search smoke test"])

    assert len(vectors) == 1
    assert len(vectors[0]) == embedder.dimensions
    assert any(v != 0.0 for v in vectors[0])
