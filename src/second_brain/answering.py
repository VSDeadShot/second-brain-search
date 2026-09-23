"""Deciding whether the model's answer may be shown.

Retrieval finds passages, generation writes prose; this module is the part that
refuses. One rule drives it: an answer nobody can trace back to a passage is not
an answer, however confident it reads. So a claimed answer that cites nothing -
or cites only passage numbers that don't exist - is turned into a refusal with a
warning, and its text is discarded rather than printed with a caveat. A caveat
is something a hurried reader skips.

Citations are resolved here, from the numbers the model returned to the chunks
they name. The model never names a file, so it cannot cite a document that isn't
in front of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .embedding import Embedder
from .freshness import DateLookup
from .generation import Generator, RawAnswer
from .retrieval import DEFAULT_ANSWER_K, RetrievedChunk, retrieve_for_answer
from .store import ChunkStore

UNCITED_WARNING = (
    "The model wrote an answer but cited no passage, so it isn't shown - there is "
    "nothing to check it against."
)


@dataclass(frozen=True)
class Citation:
    number: int
    """The passage number as the model wrote it, e.g. [1]."""
    chunk: RetrievedChunk
    date: str
    date_source: str
    changed_since_indexed: bool


@dataclass(frozen=True)
class Answer:
    question: str
    answerable: bool
    text: str
    citations: tuple[Citation, ...]
    passages: tuple[RetrievedChunk, ...]
    """Everything retrieved, cited or not - a refusal shows the closest few."""
    warnings: tuple[str, ...]
    model: str


def _refusal(
    question: str,
    passages: Sequence[RetrievedChunk],
    model: str,
    warnings: tuple[str, ...] = (),
) -> Answer:
    return Answer(
        question=question,
        answerable=False,
        text="",
        citations=(),
        passages=tuple(passages),
        warnings=warnings,
        model=model,
    )


def _resolve_citations(
    raw: RawAnswer, passages: Sequence[RetrievedChunk], dates: DateLookup
) -> tuple[tuple[Citation, ...], tuple[str, ...]]:
    kept: list[Citation] = []
    dropped: list[int] = []

    for number in dict.fromkeys(raw.citations):  # first-seen order, no duplicates
        if not 1 <= number <= len(passages):
            dropped.append(number)
            continue
        chunk = passages[number - 1]
        dated = dates.date_for(chunk.path)
        kept.append(
            Citation(
                number=number,
                chunk=chunk,
                date=dated.date,
                date_source=dated.source,
                changed_since_indexed=dates.changed_since_indexed(chunk.path, chunk.mtime),
            )
        )

    warnings: tuple[str, ...] = ()
    if dropped:
        numbers = ", ".join(str(n) for n in dropped)
        warnings = (f"The model cited passage(s) {numbers}, which it wasn't given.",)
    return tuple(sorted(kept, key=lambda c: c.number)), warnings


def answer_question(
    question: str,
    embedder: Embedder,
    store: ChunkStore,
    generator: Generator,
    *,
    k: int = DEFAULT_ANSWER_K,
    project: str | None = None,
    dates: DateLookup | None = None,
) -> Answer:
    """Retrieve, generate, and return an answer only if it is traceable.

    Raises RetrievalError before anything is embedded or generated when the
    question can't be answered as asked, and lets a GenerationError through -
    a quota message is more useful than a silent refusal.
    """
    dates = dates if dates is not None else DateLookup()
    passages = retrieve_for_answer(question, embedder, store, k=k, project=project)
    if not passages:
        return _refusal(question, (), getattr(generator, "model", "unknown"))

    raw = generator.generate(question, passages)
    model = generator.model

    if not raw.answerable or not raw.answer.strip():
        return _refusal(question, passages, model)

    citations, warnings = _resolve_citations(raw, passages, dates)
    if not citations:
        return _refusal(question, passages, model, warnings + (UNCITED_WARNING,))

    return Answer(
        question=question,
        answerable=True,
        text=raw.answer,
        citations=citations,
        passages=tuple(passages),
        warnings=warnings,
        model=model,
    )
