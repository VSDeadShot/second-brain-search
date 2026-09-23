"""The retrieval eval suite: questions with expected sources, scored mechanically.

Scoring answers one question only - did the expected projects appear in the top
k? That makes retrieval changes comparable between runs. Whether an answer is
any good stays a human judgement, so nothing here grades text.

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

SUPPORTED_VERSIONS = (1,)
_REQUIRED = ("id", "category", "text", "expected_projects")
_OPTIONAL = ("also_in_corpus", "notes")


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

    @property
    def expects_no_answer(self) -> bool:
        """No expected project means nothing in the corpus should answer it."""
        return not self.expected_projects


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

    @property
    def expects_no_answer(self) -> bool:
        return self.question.expects_no_answer


@dataclass(frozen=True)
class EvalReport:
    suite_version: int
    suite_path: Path
    generated_at: datetime
    settings: dict[str, int]
    outcomes: tuple[QuestionOutcome, ...]
    summary: dict[str, int] = field(default_factory=dict)


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

    seen.add(raw["id"])
    return EvalQuestion(
        id=raw["id"],
        category=raw["category"],
        text=raw["text"],
        expected_projects=tuple(raw["expected_projects"]),
        also_in_corpus=tuple(raw.get("also_in_corpus", ())),
        notes=str(raw.get("notes", "")).strip(),
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
    return QuestionOutcome(
        question=question,
        results=tuple(results),
        found=found,
        missing=missing,
        projects_returned=returned,
        top_score=results[0].score if results else None,
    )


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

    expecting = [o for o in outcomes if not o.expects_no_answer]
    summary = {
        "questions": len(outcomes),
        "expecting_an_answer": len(expecting),
        "all_expected_found": sum(1 for o in expecting if not o.missing),
        "any_expected_found": sum(1 for o in expecting if o.found),
    }
    return EvalReport(
        suite_version=suite.version,
        suite_path=suite.path,
        generated_at=datetime.now(timezone.utc),
        settings={"k": k, "per_file": per_file, "per_project": per_project},
        outcomes=outcomes,
        summary=summary,
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
                "expected_projects": list(o.question.expected_projects),
                "also_in_corpus": list(o.question.also_in_corpus),
                "expects_no_answer": o.expects_no_answer,
                "found": list(o.found),
                "missing": list(o.missing),
                "projects_returned": list(o.projects_returned),
                "top_score": o.top_score,
                "results": [
                    {
                        "rank": rank,
                        "project": r.project,
                        "rel_path": r.rel_path,
                        "heading_path": r.heading_path,
                        "score": r.score,
                    }
                    for rank, r in enumerate(o.results, start=1)
                ],
            }
            for o in report.outcomes
        ],
    }
