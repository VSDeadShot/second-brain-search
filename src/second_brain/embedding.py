"""Gemini embeddings.

`Embedder` is the seam the rest of the pipeline depends on, so indexing and
retrieval can be tested without a network call or an API key.

Verified live (2026-09-16): gemini-embedding-001 accepts both retrieval task
types and honours output_dimensionality. Truncated output is NOT unit length
(measured L2 norm ~0.59 at 768 dims; only the full 3072 is normalised), so
vectors are normalised here before they reach the store.

Free-tier quota, measured by probe the same day: 100 embedded TEXTS per minute
per project - every input in a batch counts, so batching gives no relief - and
a call is refused once the count is already over the limit. A full rebuild is
therefore paced at one 100-text batch per minute, and a 429 waits out the
server's own retryDelay rather than a fixed backoff.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_DIMENSIONS = 768
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_RETRIES = 3

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

FREE_TIER_ITEMS_PER_MINUTE = 100
PACING_WINDOW_SECONDS = 60
# Added to the server's retryDelay so the retry lands just after the window
# clears rather than racing its boundary.
RATE_LIMIT_MARGIN_SECONDS = 1.0

BatchCallback = Callable[[int, int], None]


class EmbeddingError(Exception):
    """The embedding backend failed, or returned something unusable."""


def _normalise(vector: list[float]) -> list[float]:
    """Scale to unit length. A zero vector is returned as-is rather than divided by 0."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def _effective_batch_size(batch_size: int, items_per_minute: int | None) -> int:
    return batch_size if items_per_minute is None else min(batch_size, items_per_minute)


def estimate_embedding_seconds(
    chunks: int,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    items_per_minute: int | None = FREE_TIER_ITEMS_PER_MINUTE,
) -> int:
    """Pacing time for a run: one full window between consecutive batches.

    Request latency is ignored - it is small, and it counts toward each window
    anyway, so the estimate is an upper bound on the waiting, not on the whole run.
    """
    if chunks <= 0 or items_per_minute is None:
        return 0
    batches = math.ceil(chunks / _effective_batch_size(batch_size, items_per_minute))
    return (batches - 1) * PACING_WINDOW_SECONDS


def _rate_limit_delay(exc: Exception) -> float | None:
    """Seconds the server asked us to wait, if `exc` is a 429; otherwise None.

    google-genai raises ClientError with `.code` and the response JSON in
    `.details`; the delay lives in a google.rpc.RetryInfo entry as e.g. "45s".
    A 429 without a parseable RetryInfo waits a full window.
    """
    if getattr(exc, "code", None) != 429:
        return None

    body = getattr(exc, "details", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    for detail in error.get("details", []) if isinstance(error, dict) else []:
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
            raw = str(detail.get("retryDelay", "")).strip().removesuffix("s")
            try:
                return float(raw)
            except ValueError:
                break
    return float(PACING_WINDOW_SECONDS)


@runtime_checkable
class Embedder(Protocol):
    @property
    def dimensions(self) -> int: ...

    def embed_documents(
        self, texts: Sequence[str], *, on_batch: BatchCallback | None = None
    ) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbedder:
    """Batches, paces to the quota, retries, and keeps vectors aligned with inputs."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = EMBEDDING_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        items_per_minute: int | None = FREE_TIER_ITEMS_PER_MINUTE,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if items_per_minute is not None and items_per_minute < 1:
            raise ValueError("items_per_minute must be at least 1, or None to disable pacing")

        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._items_per_minute = items_per_minute
        self._sleep = sleep
        self._clock = clock
        # When the most recent request attempt was sent - the start of the quota
        # window the next batch has to wait out.
        self._window_started: float | None = None
        self._client = client if client is not None else self._build_client(api_key)

    @staticmethod
    def _build_client(api_key: str | None) -> Any:
        if not api_key:
            raise EmbeddingError(
                "GEMINI_API_KEY is required to embed. Set it in .env - discovery "
                "still works without one."
            )
        # Imported lazily so the package stays importable (and testable) without
        # the SDK present.
        from google import genai

        return genai.Client(api_key=api_key)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _config(self, task_type: str) -> Any:
        from google.genai import types

        return types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self._dimensions,
        )

    def _wait_for_next_window(self) -> None:
        if self._items_per_minute is None or self._window_started is None:
            return
        remaining = PACING_WINDOW_SECONDS - (self._clock() - self._window_started)
        if remaining > 0:
            self._sleep(remaining)

    def _embed_batch(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            self._window_started = self._clock()
            try:
                response = self._client.models.embed_content(
                    model=self._model,
                    contents=list(texts),
                    config=self._config(task_type),
                )
            except Exception as exc:  # the SDK raises a family of transport errors
                last_error = exc
                if attempt < self._max_retries:
                    server_delay = _rate_limit_delay(exc)
                    if server_delay is not None:
                        self._sleep(server_delay + RATE_LIMIT_MARGIN_SECONDS)
                    else:
                        self._sleep(2 ** (attempt - 1))
                continue

            embeddings = list(response.embeddings or [])
            if len(embeddings) != len(texts):
                raise EmbeddingError(
                    f"Gemini returned {len(embeddings)} vectors, expected {len(texts)}."
                )
            return [_normalise(list(e.values)) for e in embeddings]

        raise EmbeddingError(
            f"Embedding failed after {self._max_retries} attempts: {last_error}"
        ) from last_error

    def embed_documents(
        self, texts: Sequence[str], *, on_batch: BatchCallback | None = None
    ) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []

        size = _effective_batch_size(self._batch_size, self._items_per_minute)
        batches = [items[start : start + size] for start in range(0, len(items), size)]

        vectors: list[list[float]] = []
        for number, batch in enumerate(batches, start=1):
            # Reported before the wait, so a paced minute shows which batch is queued
            # instead of looking like a hang on the previous one.
            if on_batch is not None:
                on_batch(number, len(batches))
            if number > 1:
                self._wait_for_next_window()
            vectors.extend(self._embed_batch(batch, TASK_DOCUMENT))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text], TASK_QUERY)[0]
