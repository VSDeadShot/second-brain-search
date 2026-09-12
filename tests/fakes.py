"""Test doubles. Deliberately not in the package - production has no use for them."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field


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

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@dataclass
class StubEmbedding:
    values: list[float]


@dataclass
class StubResponse:
    embeddings: list[StubEmbedding]


@dataclass
class StubModels:
    """Stands in for genai.Client().models - records calls, replays scripted results."""

    dimensions: int = 4
    failures_before_success: int = 0
    calls: list[dict] = field(default_factory=list)
    _failed: int = 0

    def embed_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        if self._failed < self.failures_before_success:
            self._failed += 1
            raise RuntimeError("transient upstream error")
        return StubResponse(
            embeddings=[StubEmbedding(values=[0.1] * self.dimensions) for _ in contents]
        )


@dataclass
class StubClient:
    models: StubModels
