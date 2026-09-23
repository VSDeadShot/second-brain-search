"""Test doubles. Deliberately not in the package - production has no use for them."""

from __future__ import annotations

import hashlib
import math
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

BatchCallback = Callable[[int, int], None]
EmbeddedCallback = Callable[[int, list[list[float]]], None]

# Quota ids exactly as the live API reported them on 2026-09-16.
PER_MINUTE_QUOTA_ID = "EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier"
PER_DAY_QUOTA_ID = "EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier"


class FakeEmbedder:
    """Deterministic offline embedder: same text always yields the same vector."""

    def __init__(self, dimensions: int = 8) -> None:
        self._dimensions = dimensions
        self.embed_calls: list[list[str]] = []

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def cache_namespace(self) -> str:
        return f"fake|{self._dimensions}"

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self._dimensions)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        on_batch: BatchCallback | None = None,
        on_embedded: EmbeddedCallback | None = None,
    ) -> list[list[float]]:
        items = list(texts)
        self.embed_calls.append(items)
        if not items:
            return []
        if on_batch is not None:
            on_batch(1, 1)
        vectors = [self._vector(t) for t in items]
        if on_embedded is not None:
            on_embedded(0, vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


_STOPWORDS = frozenset(
    "a an and are as at be by did do for from how i in is it my of on or that the "
    "this to was we what when where which with".split()
)


class KeywordEmbedder(FakeEmbedder):
    """Bag-of-words vectors: each word hashes to a fixed dimension.

    Texts sharing words land close together, so retrieval ranking is
    predictable in tests. It proves the plumbing ranks by vector similarity;
    it says nothing about how well Gemini's embeddings rank real questions.
    """

    def __init__(self, dimensions: int = 256) -> None:
        super().__init__(dimensions)
        self.query_calls: list[str] = []

    @property
    def cache_namespace(self) -> str:
        return f"keyword|{self._dimensions}"

    def _vector(self, text: str) -> list[float]:
        import re

        raw = [0.0] * self._dimensions
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word not in _STOPWORDS:
                digest = hashlib.sha256(word.encode("utf-8")).digest()
                raw[int.from_bytes(digest[:4], "big") % self._dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector(text)


class FailingEmbedder(FakeEmbedder):
    """Fails the way a quota error or bad key would: at the embed call.

    Has its own cache namespace. Sharing FakeEmbedder's would let a previous
    fake run's saved vectors satisfy every text, so the failing call would never
    be made and a test simulating an outage would silently test nothing.
    """

    @property
    def cache_namespace(self) -> str:
        return f"failing|{self._dimensions}"

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        on_batch: BatchCallback | None = None,
        on_embedded: EmbeddedCallback | None = None,
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


def rate_limit_error(
    retry_delay: str | None = "45s",
    *,
    quota_id: str = PER_MINUTE_QUOTA_ID,
    quota_value: str = "100",
) -> Exception:
    """A real google-genai ClientError, shaped like the 429s the live runs received."""
    from google.genai import errors

    details: list[dict] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
                {
                    "quotaMetric": "generativelanguage.googleapis.com/embed_content_free_tier_requests",
                    "quotaId": quota_id,
                    "quotaValue": quota_value,
                }
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


def bare_rate_limit_error() -> Exception:
    """The third live run's failure (2026-09-18, batch 3 of 9): a 429 whose only
    detail is a help link - no QuotaFailure naming the quota, no RetryInfo."""
    from google.genai import errors

    return errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota, please check your plan and billing details.",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.Help",
                        "links": [{"description": "Learn more about Gemini API quotas"}],
                    }
                ],
            }
        },
    )


def daily_quota_error() -> Exception:
    """The second live run's failure: the daily cap. Note the server still sent a
    58s retryDelay - waiting it out cannot help."""
    return rate_limit_error("58s", quota_id=PER_DAY_QUOTA_ID, quota_value="1000")


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
    `raise_on_calls` raises on specific call numbers (1-based) - e.g. a quota
    error on the third batch after two good ones.
    With a `clock`, each call's start time is recorded and `latency` seconds pass.
    """

    dimensions: int = 4
    failures_before_success: int = 0
    raise_queue: list[Exception] = field(default_factory=list)
    raise_on_calls: dict[int, Exception] = field(default_factory=dict)
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

        if len(self.calls) in self.raise_on_calls:
            raise self.raise_on_calls[len(self.calls)]
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


@dataclass
class StubGenerationResponse:
    text: str | None


@dataclass
class StubGenerationModels:
    """Stands in for genai.Client().models for generate_content."""

    text: str = '{"answerable": true, "answer": "Because [1].", "citations": [1]}'
    raises: Exception | None = None
    calls: list[dict] = field(default_factory=list)

    def generate_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.raises is not None:
            raise self.raises
        return StubGenerationResponse(text=self.text)


@dataclass
class StubGenerationClient:
    models: StubGenerationModels


def generation_daily_quota_error() -> Exception:
    """A per-day 429 for a generation model - 500/day on the Lite free tier."""
    return rate_limit_error(
        "34s",
        quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier",
        quota_value="500",
    )


class FakeGenerator:
    """A Generator that replays a scripted answer. Never touches the network."""

    def __init__(self, answer=None, *, model: str = "fake-model", raises: Exception | None = None) -> None:
        from second_brain.generation import RawAnswer

        self._answer = answer if answer is not None else RawAnswer(True, "Because [1].", (1,))
        self._model = model
        self._raises = raises
        self.prompts: list[str] = []
        self.passages: list[list] = []

    @property
    def model(self) -> str:
        return self._model

    def generate(self, question: str, passages):
        from second_brain.generation import build_prompt

        self.prompts.append(build_prompt(question, passages))
        self.passages.append(list(passages))
        if self._raises is not None:
            raise self._raises
        return self._answer


class FakeRunner:
    """Stands in for subprocess.run: records the call, replays a scripted result."""

    def __init__(self, stdout: str = "2026-08-18\n", returncode: int = 0, raises: Exception | None = None) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.raises = raises
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, "")
