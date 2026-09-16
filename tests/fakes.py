"""Test doubles. Deliberately not in the package - production has no use for them."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

BatchCallback = Callable[[int, int], None]


class FakeEmbedder:
    """Deterministic offline embedder: same text always yields the same vector."""

    def __init__(self, dimensions: int = 8) -> None:
        self._dimensions = dimensions
        self.embed_calls: list[list[str]] = []

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self._dimensions)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def embed_documents(
        self, texts: Sequence[str], *, on_batch: BatchCallback | None = None
    ) -> list[list[float]]:
        items = list(texts)
        self.embed_calls.append(items)
        if items and on_batch is not None:
            on_batch(1, 1)
        return [self._vector(t) for t in items]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FailingEmbedder(FakeEmbedder):
    """Fails the way a quota error or bad key would: at the embed call."""

    def embed_documents(
        self, texts: Sequence[str], *, on_batch: BatchCallback | None = None
    ) -> list[list[float]]:
        from second_brain.embedding import EmbeddingError

        raise EmbeddingError("simulated upstream failure")


class FakeClock:
    """Monotonic clock whose sleep advances time instantly and records the wait."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def rate_limit_error(retry_delay: str | None = "45s") -> Exception:
    """A real google-genai ClientError, shaped like the 429 the live run received."""
    from google.genai import errors

    details: list[dict] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
                {"quotaMetric": "generativelanguage.googleapis.com/embed_content_free_tier_requests"}
            ],
        }
    ]
    if retry_delay is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})

    return errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota.",
                "status": "RESOURCE_EXHAUSTED",
                "details": details,
            }
        },
    )


@dataclass
class StubEmbedding:
    values: list[float]


@dataclass
class StubResponse:
    embeddings: list[StubEmbedding]


@dataclass
class StubModels:
    """Stands in for genai.Client().models - records calls, replays scripted results.

    `raise_queue` errors are raised one per call, in order, before any success.
    With a `clock`, each call's start time is recorded and `latency` seconds pass.
    """

    dimensions: int = 4
    failures_before_success: int = 0
    raise_queue: list[Exception] = field(default_factory=list)
    clock: FakeClock | None = None
    latency: float = 0.0
    calls: list[dict] = field(default_factory=list)
    call_times: list[float] = field(default_factory=list)
    _failed: int = 0

    def embed_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        if self.clock is not None:
            self.call_times.append(self.clock.now)
            self.clock.now += self.latency

        if self.raise_queue:
            raise self.raise_queue.pop(0)
        if self._failed < self.failures_before_success:
            self._failed += 1
            raise RuntimeError("transient upstream error")
        return StubResponse(
            embeddings=[StubEmbedding(values=[0.1] * self.dimensions) for _ in contents]
        )


@dataclass
class StubClient:
    models: StubModels
