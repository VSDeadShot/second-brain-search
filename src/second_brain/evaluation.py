"""The retrieval eval suite: questions with expected sources, scored mechanically.

Each question is scored by one of three checks, chosen by the fields it carries:

- text: a passage in the top k must contain one of `expected_text`. This is the
  check that matters for a fact. Q1 showed why: the right project came back, no
  passage named the model, `sbs ask` refused - and a project-level check still
  called it a pass.
- project: every `expected_projects` entry must appear in the top k. Right for
  "which of my projects..." questions, where the project IS the answer.
- no-answer: nothing in the corpus answers it. Scores are kept so a weak match
  stays visible, but nothing is passed or failed; whether `sbs ask` refuses is
  checked by asking, since this suite never generates.

Scoring stays mechanical so retrieval changes are comparable between runs.
Whether an answer is any good stays a human judgement.

Running a suite embeds one query per question and reads the index; it never
generates anything.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .embedding import Embedder
from .retrieval import (
    DEFAULT_ANSWER_K,
    DEFAULT_PER_FILE,
    DEFAULT_PER_PROJECT,
    RetrievedChunk,
    retrieve_for_answer,
)
from .store import ChunkStore

# v2 split Q1 into a fact and a rationale question and added the text check.
# Version 1 is refused: scored by v2 rules it would produce a report that looks
# comparable with old v1 runs and isn't.
SUPPORTED_VERSIONS = (2,)
_REQUIRED = ("id", "category", "text", "expected_projects")
_OPTIONAL = ("also_in_corpus", "notes", "expected_text")

CHECK_TEXT = "text"
CHECK_PROJECT = "project"
CHECK_NO_ANSWER = "no-answer"


class EvalSuiteError(Exception):
    """The suite file is missing, unreadable, or malformed."""


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    category: str
    text: str
    expected_projects: tuple[str, ...]
    also_in_corpus: tuple[str, ...] = ()
    notes: str = ""
    expected_text: tuple[str, ...] = ()

    @property
    def expects_no_answer(self) -> bool:
        """No expected project means nothing in the corpus should answer it."""
        return not self.expected_projects

    @property
    def check(self) -> str:
        if self.expects_no_answer:
            return CHECK_NO_ANSWER
        return CHECK_TEXT if self.expected_text else CHECK_PROJECT


@dataclass(frozen=True)
class TextMatch:
    """Where an expected string was first found."""

    text: str
    rank: int
    project: str
    rel_path: str
    chunk_index: int


@dataclass(frozen=True)
class EvalSuite:
    version: int
    path: Path
    questions: tuple[EvalQuestion, ...]


@dataclass(frozen=True)
class QuestionOutcome:
    question: EvalQuestion
    results: tuple[RetrievedChunk, ...]
    found: tuple[str, ...]
    missing: tuple[str, ...]
    projects_returned: tuple[str, ...]
    top_score: float | None
    passed: bool | None = None
    """None for a no-answer question, which is neither passed nor failed."""
    reason: str = ""
    text_matches: tuple[TextMatch, ...] = ()

    @property
    def expects_no_answer(self) -> bool:
        return self.question.expects_no_answer

    @property
    def check(self) -> str:
        return self.question.check


@dataclass(frozen=True)
class EvalReport:
    suite_version: int
    suite_path: Path
    generated_at: datetime
    settings: dict[str, int]
    outcomes: tuple[QuestionOutcome, ...]
    summary: dict[str, Any] = field(default_factory=dict)


def _expected_text(raw: dict[str, Any], where: str) -> tuple[str, ...]:
    if "expected_text" not in raw:
        return ()
    value = raw["expected_text"]
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(v, str) and v.strip() for v in value)
    ):
        raise EvalSuiteError(f"{where}: expected_text must be a non-empty list of non-blank strings")
    if not raw["expected_projects"]:
        raise EvalSuiteError(
            f"{where}: expected_text on a question that expects no answer - it can't "
            "expect nothing and expect this text"
        )
    return tuple(v.strip() for v in value)


def _question_from(raw: dict[str, Any], index: int, seen: set[str]) -> EvalQuestion:
    where = raw.get("id") or f"question #{index + 1}"

    unknown = sorted(set(raw) - set(_REQUIRED) - set(_OPTIONAL))
    if unknown:
        raise EvalSuiteError(f"{where}: unknown key(s): {', '.join(unknown)}")
    missing = [key for key in _REQUIRED if key not in raw]
    if missing:
        raise EvalSuiteError(f"{where}: missing required key(s): {', '.join(missing)}")
    if raw["id"] in seen:
        raise EvalSuiteError(f"duplicate question id: {raw['id']}")

    for key in ("expected_projects", *(k for k in ("also_in_corpus",) if k in raw)):
        value = raw[key]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise EvalSuiteError(f"{where}: {key} must be a list of strings")

    expected_text = _expected_text(raw, where)

    seen.add(raw["id"])
    return EvalQuestion(
        id=raw["id"],
        category=raw["category"],
        text=raw["text"],
        expected_projects=tuple(raw["expected_projects"]),
        also_in_corpus=tuple(raw.get("also_in_corpus", ())),
        notes=str(raw.get("notes", "")).strip(),
        expected_text=expected_text,
    )


def load_eval_suite(path: Path) -> EvalSuite:
    path = Path(path)
    if not path.is_file():
        raise EvalSuiteError(f"Eval suite not found: {path}")

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise EvalSuiteError(f"{path.name} is not valid TOML: {exc}") from exc

    version = raw.get("version")
    if version not in SUPPORTED_VERSIONS:
        raise EvalSuiteError(
            f"{path.name}: unsupported suite version {version!r}; "
            f"supported: {', '.join(str(v) for v in SUPPORTED_VERSIONS)}"
        )

    seen: set[str] = set()
    questions = tuple(
        _question_from(item, i, seen) for i, item in enumerate(raw.get("question", []))
    )
    if not questions:
        raise EvalSuiteError(f"{path.name}: no [[question]] entries")
    return EvalSuite(version=version, path=path, questions=questions)


def _normalise(text: str) -> str:
    """Case-folded, with every run of whitespace collapsed to one space.

    The corpus is hard-wrapped, so a phrase can break across lines anywhere. This
    is the only loosening: a near miss - a prefix, spaces for hyphens - still
    doesn't match.
    """
    return " ".join(text.split()).casefold()


def _text_matches(
    expected: Sequence[str], results: Sequence[RetrievedChunk]
) -> tuple[TextMatch, ...]:
    """The first passage containing each expected string.

    Only the passage text counts. A heading that names the fact is not a passage
    stating it.
    """
    haystacks = [_normalise(r.text) for r in results]
    matches = []
    for needle in expected:
        wanted = _normalise(needle)
        for rank, (result, haystack) in enumerate(zip(results, haystacks), start=1):
            if wanted in haystack:
                matches.append(
                    TextMatch(needle, rank, result.project, result.rel_path, result.chunk_index)
                )
                break
    return tuple(sorted(matches, key=lambda m: m.rank))


def evaluate_question(
    question: EvalQuestion, results: Sequence[RetrievedChunk]
) -> QuestionOutcome:
    returned = tuple(dict.fromkeys(r.project for r in results))  # first-seen order
    found = () if question.expects_no_answer else tuple(
        p for p in question.expected_projects if p in returned
    )
    missing = () if question.expects_no_answer else tuple(
        p for p in question.expected_projects if p not in returned
    )
    top_score = results[0].score if results else None
    matches: tuple[TextMatch, ...] = ()

    if question.check == CHECK_NO_ANSWER:
        passed = None
        best = f"top score {top_score:.3f}" if top_score is not None else "nothing retrieved"
        reason = f"expects no answer; {best}"
    elif question.check == CHECK_TEXT:
        matches = _text_matches(question.expected_text, results)
        passed = bool(matches)
        if matches:
            m = matches[0]
            reason = f"{m.text!r} at rank {m.rank} ({m.project} / {m.rel_path} #{m.chunk_index})"
        else:
            # Say whether the project came back: "right place, no fact" is Q1's
            # failure, and it reads very differently from "wrong place".
            where = (
                f"{', '.join(found)} returned" if found else f"{', '.join(missing)} not returned"
            )
            reason = f"none of {list(question.expected_text)} in the top {len(results)} ({where})"
    else:
        passed = not missing
        reason = f"found {', '.join(found)}" if passed else f"missing {', '.join(missing)}"
        if missing and found:
            reason += f"; found {', '.join(found)}"

    return QuestionOutcome(
        question=question,
        results=tuple(results),
        found=found,
        missing=missing,
        projects_returned=returned,
        top_score=top_score,
        passed=passed,
        reason=reason,
        text_matches=matches,
    )


def _summarise(outcomes: Sequence[QuestionOutcome]) -> dict[str, Any]:
    expecting = [o for o in outcomes if not o.expects_no_answer]
    by_check = {
        check: {
            "scored": sum(1 for o in outcomes if o.check == check),
            "passed": sum(1 for o in outcomes if o.check == check and o.passed),
        }
        for check in (CHECK_TEXT, CHECK_PROJECT)
    }
    return {
        "questions": len(outcomes),
        "passed": sum(1 for o in outcomes if o.passed is True),
        "failed": sum(1 for o in outcomes if o.passed is False),
        "no_answer": sum(1 for o in outcomes if o.passed is None),
        "by_check": by_check,
        # Project-level counts, kept so v2 reports can be read beside v1 ones.
        "expecting_an_answer": len(expecting),
        "all_expected_found": sum(1 for o in expecting if not o.missing),
        "any_expected_found": sum(1 for o in expecting if o.found),
    }


def run_eval(
    suite: EvalSuite,
    embedder: Embedder,
    store: ChunkStore,
    *,
    k: int = DEFAULT_ANSWER_K,
    per_file: int = DEFAULT_PER_FILE,
    per_project: int = DEFAULT_PER_PROJECT,
) -> EvalReport:
    outcomes = tuple(
        evaluate_question(
            question,
            retrieve_for_answer(
                question.text, embedder, store, k=k, per_file=per_file, per_project=per_project
            ),
        )
        for question in suite.questions
    )
    return EvalReport(
        suite_version=suite.version,
        suite_path=suite.path,
        generated_at=datetime.now(timezone.utc),
        settings={"k": k, "per_file": per_file, "per_project": per_project},
        outcomes=outcomes,
        summary=_summarise(outcomes),
    )


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    return {
        "suite_version": report.suite_version,
        "suite_path": str(report.suite_path),
        "generated_at": report.generated_at.isoformat(),
        "settings": report.settings,
        "summary": report.summary,
        "questions": [
            {
                "id": o.question.id,
                "category": o.question.category,
                "text": o.question.text,
                "check": o.check,
                "passed": o.passed,
                "reason": o.reason,
                "expected_projects": list(o.question.expected_projects),
                "expected_text": list(o.question.expected_text),
                "also_in_corpus": list(o.question.also_in_corpus),
                "expects_no_answer": o.expects_no_answer,
                "found": list(o.found),
                "missing": list(o.missing),
                "projects_returned": list(o.projects_returned),
                "top_score": o.top_score,
                "text_matches": [
                    {
                        "text": m.text,
                        "rank": m.rank,
                        "project": m.project,
                        "rel_path": m.rel_path,
                        "chunk_index": m.chunk_index,
                    }
                    for m in o.text_matches
                ],
                "results": [
                    {
                        "rank": rank,
                        "project": r.project,
                        "rel_path": r.rel_path,
                        "heading_path": r.heading_path,
                        "chunk_index": r.chunk_index,
                        "score": r.score,
                    }
                    for rank, r in enumerate(o.results, start=1)
                ],
            }
            for o in report.outcomes
        ],
    }
