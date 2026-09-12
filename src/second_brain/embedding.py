"""Gemini embeddings.

`Embedder` is the seam the rest of the pipeline depends on, so indexing and
retrieval can be tested without a network call or an API key.

NOTE: the live call is unverified as of slice 2 - no API key was available.
`test_live_gemini_embedding_roundtrip` exercises it once a key is set, and
EMBEDDING_MODEL is the single place to correct the name if it has moved on.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_DIMENSIONS = 768
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_RETRIES = 3

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


class EmbeddingError(Exception):
    """The embedding backend failed, or returned something unusable."""


@runtime_checkable
class Embedder(Protocol):
    @property
    def dimensions(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbedder:
    """Batches, retries with backoff, and keeps vectors aligned with inputs."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = EMBEDDING_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._sleep = sleep
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

    def _embed_batch(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.models.embed_content(
                    model=self._model,
                    contents=list(texts),
                    config=self._config(task_type),
                )
            except Exception as exc:  # the SDK raises a family of transport errors
                last_error = exc
                if attempt < self._max_retries:
                    self._sleep(2 ** (attempt - 1))
                continue

            embeddings = list(response.embeddings or [])
            if len(embeddings) != len(texts):
                raise EmbeddingError(
                    f"Gemini returned {len(embeddings)} vectors, expected {len(texts)}."
                )
            return [list(e.values) for e in embeddings]

        raise EmbeddingError(
            f"Embedding failed after {self._max_retries} attempts: {last_error}"
        ) from last_error

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(items), self._batch_size):
            vectors.extend(
                self._embed_batch(items[start : start + self._batch_size], TASK_DOCUMENT)
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text], TASK_QUERY)[0]
