"""Embedding: batching, retry, and the shape of what comes back."""

from __future__ import annotations

import math
import os

import pytest

from second_brain.embedding import (
    FREE_TIER_ITEMS_PER_MINUTE,
    PACING_WINDOW_SECONDS,
    RATE_LIMIT_MARGIN_SECONDS,
    EmbeddingError,
    GeminiEmbedder,
    estimate_embedding_seconds,
)

from fakes import (
    FakeClock,
    FakeEmbedder,
    StubClient,
    StubEmbedding,
    StubModels,
    StubResponse,
    rate_limit_error,
)


def l2_norm(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


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


def test_document_vectors_are_unit_length() -> None:
    """Truncated gemini-embedding-001 output is not unit length (measured L2 ~0.59
    at 768 dims). Cosine ranking tolerates that, but L2 or inner-product search
    would silently mis-rank - so normalise at the boundary."""
    models = StubModels(dimensions=4)  # stub returns [0.1] * 4, L2 norm 0.2

    vectors = make_embedder(models).embed_documents(["a", "b"])

    assert all(l2_norm(v) == pytest.approx(1.0) for v in vectors)


def test_query_vector_is_unit_length() -> None:
    models = StubModels(dimensions=4)

    vector = make_embedder(models).embed_query("a")

    assert l2_norm(vector) == pytest.approx(1.0)


def test_normalisation_preserves_direction() -> None:
    models = StubModels(dimensions=3)
    models.embed_content = lambda **kw: StubResponse(
        embeddings=[StubEmbedding(values=[3.0, 4.0, 0.0])]
    )

    vector = make_embedder(models, dimensions=3).embed_documents(["a"])[0]

    assert vector == pytest.approx([0.6, 0.8, 0.0])


def test_zero_vector_is_returned_without_dividing_by_zero() -> None:
    models = StubModels(dimensions=4)
    models.embed_content = lambda **kw: StubResponse(
        embeddings=[StubEmbedding(values=[0.0] * 4)]
    )

    vector = make_embedder(models).embed_documents(["a"])[0]

    assert vector == [0.0] * 4


def test_fake_embedder_is_deterministic() -> None:
    fake = FakeEmbedder()
    assert fake.embed_documents(["x"])[0] == fake.embed_documents(["x"])[0]
    assert fake.embed_query("x") == fake.embed_documents(["x"])[0]


# --- free-tier pacing -----------------------------------------------------------
#
# Measured against the live API (2026-09-16): the free tier allows 100 embedded
# TEXTS per minute - each input in a batch counts - and refuses a call once the
# count is already over the limit. Batching alone gives no quota relief.


def paced_embedder(models: StubModels, clock: FakeClock, **kwargs) -> GeminiEmbedder:
    models.clock = clock
    return make_embedder(models, sleep=clock.sleep, clock=clock.monotonic, **kwargs)


def texts(n: int) -> list[str]:
    return [f"t{i}" for i in range(n)]


def test_free_tier_limit_is_100_texts_per_minute() -> None:
    assert FREE_TIER_ITEMS_PER_MINUTE == 100
    assert PACING_WINDOW_SECONDS == 60


def test_batches_never_exceed_the_per_minute_text_limit() -> None:
    models, clock = StubModels(), FakeClock()
    embedder = paced_embedder(models, clock, batch_size=250, items_per_minute=100)

    embedder.embed_documents(texts(250))

    assert [len(c["contents"]) for c in models.calls] == [100, 100, 50]


def test_each_batch_waits_for_the_rest_of_the_minute() -> None:
    models, clock = StubModels(), FakeClock()
    embedder = paced_embedder(models, clock, items_per_minute=100)

    embedder.embed_documents(texts(250))

    assert models.call_times == [0.0, 60.0, 120.0]


def test_request_time_counts_toward_the_minute() -> None:
    """A batch that takes 5s to return leaves 55s to wait, not 60."""
    models, clock = StubModels(latency=5.0), FakeClock()
    embedder = paced_embedder(models, clock, items_per_minute=100)

    embedder.embed_documents(texts(200))

    assert models.call_times == [0.0, 60.0]
    assert clock.sleeps == [55.0]


def test_a_single_batch_never_waits() -> None:
    models, clock = StubModels(), FakeClock()
    embedder = paced_embedder(models, clock, items_per_minute=100)

    embedder.embed_documents(texts(100))

    assert clock.sleeps == []


def test_pacing_can_be_disabled_for_paid_tiers() -> None:
    models, clock = StubModels(), FakeClock()
    embedder = paced_embedder(models, clock, batch_size=100, items_per_minute=None)

    embedder.embed_documents(texts(250))

    assert [len(c["contents"]) for c in models.calls] == [100, 100, 50]
    assert clock.sleeps == []


# --- 429 handling ---------------------------------------------------------------


def test_rate_limit_waits_the_servers_retry_delay() -> None:
    models, clock = StubModels(raise_queue=[rate_limit_error("45s")]), FakeClock()
    embedder = paced_embedder(models, clock)

    vectors = embedder.embed_documents(texts(1))

    assert len(vectors) == 1
    assert clock.sleeps == [45.0 + RATE_LIMIT_MARGIN_SECONDS]


def test_rate_limit_delay_may_be_fractional() -> None:
    """The live 429 said 'retryDelay': '45s' but its message said 45.676528856s."""
    models, clock = StubModels(raise_queue=[rate_limit_error("45.676528856s")]), FakeClock()

    paced_embedder(models, clock).embed_documents(texts(1))

    assert clock.sleeps == [pytest.approx(45.676528856 + RATE_LIMIT_MARGIN_SECONDS)]


def test_rate_limit_without_retry_info_waits_a_full_window() -> None:
    models, clock = StubModels(raise_queue=[rate_limit_error(retry_delay=None)]), FakeClock()

    paced_embedder(models, clock).embed_documents(texts(1))

    assert clock.sleeps == [PACING_WINDOW_SECONDS + RATE_LIMIT_MARGIN_SECONDS]


def test_other_errors_keep_exponential_backoff() -> None:
    models, clock = StubModels(failures_before_success=2), FakeClock()

    paced_embedder(models, clock, max_retries=3).embed_documents(texts(1))

    assert clock.sleeps == [1, 2]


def test_a_retried_batch_restarts_the_pacing_window() -> None:
    """After a 46s rate-limit wait, the next batch counts its minute from the retry."""
    models, clock = StubModels(raise_queue=[rate_limit_error("45s")]), FakeClock()
    embedder = paced_embedder(models, clock, items_per_minute=100)

    embedder.embed_documents(texts(200))

    retry_at = 45.0 + RATE_LIMIT_MARGIN_SECONDS
    assert models.call_times == [0.0, retry_at, retry_at + 60.0]


def test_persistent_rate_limiting_gives_up_with_the_quota_message() -> None:
    models = StubModels(raise_queue=[rate_limit_error() for _ in range(3)])
    embedder = paced_embedder(models, FakeClock(), max_retries=3)

    with pytest.raises(EmbeddingError, match="429"):
        embedder.embed_documents(texts(1))


# --- progress -------------------------------------------------------------------


def test_progress_is_reported_before_each_batch() -> None:
    models, clock = StubModels(), FakeClock()
    seen: list[tuple[int, int]] = []

    paced_embedder(models, clock, items_per_minute=100).embed_documents(
        texts(250), on_batch=lambda n, total: seen.append((n, total))
    )

    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_no_progress_for_empty_input() -> None:
    seen: list[tuple[int, int]] = []

    make_embedder(StubModels()).embed_documents([], on_batch=lambda n, t: seen.append((n, t)))

    assert seen == []


# --- time estimate --------------------------------------------------------------


@pytest.mark.parametrize(
    ("chunks", "expected_seconds"),
    [
        (0, 0),
        (1, 0),
        (100, 0),  # one batch, no wait
        (101, 60),  # two batches, one wait
        (812, 480),  # the real corpus: 9 batches, 8 waits
    ],
)
def test_estimate_counts_one_minute_per_wait_between_batches(
    chunks: int, expected_seconds: int
) -> None:
    assert estimate_embedding_seconds(chunks) == expected_seconds


def test_estimate_is_zero_when_pacing_is_disabled() -> None:
    assert estimate_embedding_seconds(812, items_per_minute=None) == 0


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
    # The real API returns non-unit vectors at 768 dims; this proves the
    # normalisation holds against actual output, not just the stub.
    assert l2_norm(vectors[0]) == pytest.approx(1.0)
