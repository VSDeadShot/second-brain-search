"""Gemini embeddings.

`Embedder` is the seam the rest of the pipeline depends on, so indexing and
retrieval can be tested without a network call or an API key.

Verified live (2026-09-16): gemini-embedding-001 accepts both retrieval task
types and honours output_dimensionality. Truncated output is NOT unit length
(measured L2 norm ~0.59 at 768 dims; only the full 3072 is normalised), so
vectors are normalised here before they reach the store.

Free-tier limits for gemini-embedding-001 ("Gemini Embedding 1"), read from
AI Studio's rate-limit page for this project on 2026-09-18:
- RPM 100 - embedded TEXTS, not HTTP calls: every input in a batch counts, and
  a call is refused once the count is already over. (Probed 2026-09-16.)
- RPD 1000 texts, resetting at midnight Pacific.
- TPM 30,000 input tokens. Exceeding it returns a bare 429 - no quota named,
  no retry delay. A lone 30.1k batch could never pass; and two consecutive
  ~20k batches a minute apart were refused, so neighbouring batches count
  against the same window.
So a batch closes at 100 texts or 12k estimated tokens, one batch per minute:
two back-to-back batches stay at most 24k, 20% under the 30k limit. The
chars / 4 estimate was checked against count_tokens: within 5%.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from .gemini_errors import (
    RATE_LIMIT_MARGIN_SECONDS,
    RATE_WINDOW_SECONDS,
    daily_quota_limit,
    is_unexplained_rate_limit,
    rate_limit_delay,
)

EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_DIMENSIONS = 768
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_RETRIES = 3

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

FREE_TIER_ITEMS_PER_MINUTE = 100
FREE_TIER_TOKENS_PER_MINUTE = 30_000
# Two consecutive batches can land in one rate window, so each batch gets at
# most 40% of the limit: back-to-back batches then total <= 80%.
TOKEN_BUDGET_PER_MINUTE = FREE_TIER_TOKENS_PER_MINUTE * 2 // 5  # 12,000
CHARS_PER_TOKEN_ESTIMATE = 4

BatchCallback = Callable[[int, int], None]
# (offset into the input texts, normalised vectors for that batch)
EmbeddedCallback = Callable[[int, list[list[float]]], None]


class EmbeddingError(Exception):
    """The embedding backend failed, or returned something unusable."""


class DailyQuotaExceeded(EmbeddingError):
    """The per-day quota is used up. Retrying within the day cannot succeed."""


class UnexplainedRateLimit(EmbeddingError):
    """A 429 naming no quota and giving no retry delay. Waiting didn't help live,
    so it fails at once instead of burning minutes on doomed retries."""


def gemini_cache_namespace(model: str, dimensions: int) -> str:
    """Identifies a vector space: saved vectors are only reusable within one."""
    return f"{model}|{dimensions}|{TASK_DOCUMENT}"


def estimate_tokens(text: str) -> int:
    """Rough token count for batch sizing - chars / 4, rounded up. No API call."""
    return math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)


def plan_batches(
    texts: Sequence[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    items_per_minute: int | None = FREE_TIER_ITEMS_PER_MINUTE,
    tokens_per_minute: int | None = TOKEN_BUDGET_PER_MINUTE,
) -> list[list[str]]:
    """Split texts, in order, into batches that respect both the count cap and the
    token budget. A single text over the budget still goes, alone - never dropped."""
    max_count = batch_size if items_per_minute is None else min(batch_size, items_per_minute)

    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        tokens = estimate_tokens(text)
        too_many = len(current) >= max_count
        too_big = (
            tokens_per_minute is not None
            and current
            and current_tokens + tokens > tokens_per_minute
        )
        if too_many or too_big:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(text)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def estimate_embedding_seconds(
    texts: Sequence[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    items_per_minute: int | None = FREE_TIER_ITEMS_PER_MINUTE,
    tokens_per_minute: int | None = TOKEN_BUDGET_PER_MINUTE,
) -> int:
    """Pacing time for a run: one full window between consecutive batches.

    Request latency is ignored - it is small, and it counts toward each window
    anyway, so the estimate is an upper bound on the waiting, not on the whole run.
    """
    if items_per_minute is None:
        return 0
    batches = plan_batches(
        texts,
        batch_size=batch_size,
        items_per_minute=items_per_minute,
        tokens_per_minute=tokens_per_minute,
    )
    return max(len(batches) - 1, 0) * RATE_WINDOW_SECONDS


def _normalise(vector: list[float]) -> list[float]:
    """Scale to unit length. A zero vector is returned as-is rather than divided by 0."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


@runtime_checkable
class Embedder(Protocol):
    @property
    def dimensions(self) -> int: ...

    @property
    def cache_namespace(self) -> str: ...

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        on_batch: BatchCallback | None = None,
        on_embedded: EmbeddedCallback | None = None,
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
        tokens_per_minute: int | None = TOKEN_BUDGET_PER_MINUTE,
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
        if tokens_per_minute is not None and tokens_per_minute < 1:
            raise ValueError("tokens_per_minute must be at least 1, or None to disable")

        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._items_per_minute = items_per_minute
        self._tokens_per_minute = tokens_per_minute
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

    @property
    def cache_namespace(self) -> str:
        return gemini_cache_namespace(self._model, self._dimensions)

    def _config(self, task_type: str) -> Any:
        from google.genai import types

        return types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self._dimensions,
        )

    def _wait_for_next_window(self) -> None:
        if self._items_per_minute is None or self._window_started is None:
            return
        remaining = RATE_WINDOW_SECONDS - (self._clock() - self._window_started)
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
                daily_limit = daily_quota_limit(exc)
                if daily_limit is not None:
                    raise DailyQuotaExceeded(
                        f"Gemini's free-tier daily limit of {daily_limit} embedded texts is "
                        "used up. It resets at midnight Pacific time - run `sbs index` again "
                        "after that."
                    ) from exc
                if is_unexplained_rate_limit(exc):
                    tokens = sum(estimate_tokens(t) for t in texts)
                    raise UnexplainedRateLimit(
                        f"Gemini refused a batch of {len(texts)} texts (~{tokens} estimated "
                        "tokens) with a 429 that names no quota and gives no retry delay. "
                        "Waiting didn't clear this before, so the run stops here."
                    ) from exc
                last_error = exc
                if attempt < self._max_retries:
                    server_delay = rate_limit_delay(exc)
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
        self,
        texts: Sequence[str],
        *,
        on_batch: BatchCallback | None = None,
        on_embedded: EmbeddedCallback | None = None,
    ) -> list[list[float]]:
        """`on_embedded(offset, vectors)` fires as each batch returns, so a caller
        can save results before a later batch fails."""
        items = list(texts)
        if not items:
            return []

        batches = plan_batches(
            items,
            batch_size=self._batch_size,
            items_per_minute=self._items_per_minute,
            tokens_per_minute=self._tokens_per_minute,
        )

        vectors: list[list[float]] = []
        for number, batch in enumerate(batches, start=1):
            # Reported before the wait, so a paced minute shows which batch is queued
            # instead of looking like a hang on the previous one.
            if on_batch is not None:
                on_batch(number, len(batches))
            if number > 1:
                self._wait_for_next_window()
            batch_vectors = self._embed_batch(batch, TASK_DOCUMENT)
            if on_embedded is not None:
                on_embedded(len(vectors), batch_vectors)
            vectors.extend(batch_vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text], TASK_QUERY)[0]
