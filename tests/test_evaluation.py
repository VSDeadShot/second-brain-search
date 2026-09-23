"""The retrieval eval suite: a versioned set of questions with expected sources.

Scoring is deliberately mechanical - did the expected projects appear in the top
k - so retrieval changes can be compared run to run. Whether an answer is any
good stays a human judgement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from second_brain.config import Config, DEFAULT_EXCLUDE_DIRS, DEFAULT_INCLUDE_PATTERNS
from second_brain.evaluation import (
    EvalSuiteError,
    evaluate_question,
    load_eval_suite,
    report_to_dict,
    run_eval,
)
from second_brain.pipeline import collect_chunks, store_chunks
from second_brain.retrieval import RetrievedChunk
from second_brain.store import ChunkStore

from fakes import KeywordEmbedder

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_SUITE = REPO_ROOT / "eval" / "retrieval_v1.toml"

VALID = """
version = 1

[[question]]
id = "q1"
category = "exact-fact"
text = "Which model does Interview Flashcards use?"
expected_projects = ["Interview Flashcards"]

[[question]]
id = "q2"
category = "trap"
text = "How did I set up Kubernetes?"
expected_projects = []
notes = "nothing in the corpus should answer this"
"""


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "suite.toml"
    path.write_text(body, encoding="utf-8")
    return path


def chunk(project: str, rel_path: str = "README.md", score: float = 0.7) -> RetrievedChunk:
    return RetrievedChunk(
        project=project,
        rel_path=rel_path,
        heading_path="H",
        chunk_index=0,
        text="t",
        score=score,
        content_hash="h",
        mtime=1.0,
        path=f"/{project}/{rel_path}",
    )


# --- loading the suite ----------------------------------------------------------


def test_loads_questions_in_file_order(tmp_path: Path) -> None:
    suite = load_eval_suite(write(tmp_path, VALID))

    assert [q.id for q in suite.questions] == ["q1", "q2"]
    assert suite.questions[0].expected_projects == ("Interview Flashcards",)
    assert suite.questions[1].notes == "nothing in the corpus should answer this"


def test_a_question_with_no_expected_projects_expects_no_answer(tmp_path: Path) -> None:
    suite = load_eval_suite(write(tmp_path, VALID))

    assert suite.questions[0].expects_no_answer is False
    assert suite.questions[1].expects_no_answer is True


def test_missing_required_field_names_the_question(tmp_path: Path) -> None:
    body = '\nversion = 1\n\n[[question]]\nid = "q1"\ncategory = "x"\n'

    with pytest.raises(EvalSuiteError, match="q1"):
        load_eval_suite(write(tmp_path, body))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    body = VALID + '\n[[question]]\nid = "q1"\ncategory = "x"\ntext = "t"\nexpected_projects = []\n'

    with pytest.raises(EvalSuiteError, match="q1"):
        load_eval_suite(write(tmp_path, body))


def test_an_unknown_version_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvalSuiteError, match="version"):
        load_eval_suite(write(tmp_path, VALID.replace("version = 1", "version = 99")))


def test_a_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(EvalSuiteError, match="not found"):
        load_eval_suite(tmp_path / "nope.toml")


def test_the_committed_suite_holds_the_ten_eval_questions() -> None:
    suite = load_eval_suite(COMMITTED_SUITE)

    ids = [q.id for q in suite.questions]
    assert len(ids) == 10
    assert len(set(ids)) == 10
    assert all(q.text and q.category for q in suite.questions)
    # Q10 (Kubernetes) is the trap: nothing in the corpus answers it.
    assert suite.questions[9].expects_no_answer is True
    # Q5's fix lives only in git history, which the index does not cover.
    assert "git" in suite.questions[4].notes.lower()


# --- scoring --------------------------------------------------------------------


def test_expected_projects_present_are_found() -> None:
    outcome = evaluate_question(
        _question(expected=("Interview Flashcards", "DSA Tracker")),
        [chunk("Interview Flashcards"), chunk("Now Brief")],
    )

    assert outcome.found == ("Interview Flashcards",)
    assert outcome.missing == ("DSA Tracker",)


def test_outcome_records_the_top_score_and_projects_seen() -> None:
    outcome = evaluate_question(
        _question(expected=("A",)), [chunk("B", score=0.66), chunk("C", score=0.61)]
    )

    assert outcome.top_score == 0.66
    assert outcome.projects_returned == ("B", "C")
    assert outcome.found == ()
    assert outcome.missing == ("A",)


def test_a_no_answer_question_scores_nothing_but_keeps_its_scores() -> None:
    outcome = evaluate_question(_question(expected=()), [chunk("B", score=0.64)])

    assert outcome.expects_no_answer is True
    assert (outcome.found, outcome.missing) == ((), ())
    assert outcome.top_score == 0.64


def test_no_results_leaves_the_top_score_unset() -> None:
    outcome = evaluate_question(_question(expected=("A",)), [])

    assert outcome.top_score is None
    assert outcome.missing == ("A",)


def _question(*, expected: tuple[str, ...]):
    from second_brain.evaluation import EvalQuestion

    return EvalQuestion(
        id="qx", category="c", text="t", expected_projects=expected, also_in_corpus=(), notes=""
    )


# --- running the suite ----------------------------------------------------------


def config_for(root: Path) -> Config:
    return Config(
        scan_root=root,
        gemini_api_key=None,
        include_patterns=DEFAULT_INCLUDE_PATTERNS,
        include_dirs=("docs",),
        exclude_dirs=DEFAULT_EXCLUDE_DIRS,
        exclude_paths=(),
    )


@pytest.fixture
def indexed_knowledge(tmp_path: Path, knowledge: Path):
    store, embedder = ChunkStore(tmp_path / "chroma"), KeywordEmbedder()
    store_chunks(collect_chunks(config_for(knowledge)), embedder, store, rebuild=True)
    return store, embedder


def test_run_eval_embeds_one_query_per_question(tmp_path: Path, indexed_knowledge) -> None:
    store, embedder = indexed_knowledge
    suite = load_eval_suite(write(tmp_path, VALID))

    report = run_eval(suite, embedder, store, k=5)

    assert len(report.outcomes) == 2
    assert len(embedder.query_calls) == 2


def test_run_eval_records_the_settings_it_used(tmp_path: Path, indexed_knowledge) -> None:
    store, embedder = indexed_knowledge
    suite = load_eval_suite(write(tmp_path, VALID))

    report = run_eval(suite, embedder, store, k=7, per_file=2, per_project=4)

    assert report.settings == {"k": 7, "per_file": 2, "per_project": 4}


def test_report_summary_counts_questions(tmp_path: Path, indexed_knowledge) -> None:
    store, embedder = indexed_knowledge
    suite = load_eval_suite(write(tmp_path, VALID))

    summary = run_eval(suite, embedder, store, k=5).summary

    assert summary["questions"] == 2
    assert summary["expecting_an_answer"] == 1
    assert set(summary) >= {"questions", "expecting_an_answer", "all_expected_found"}


def test_report_serialises_to_json(tmp_path: Path, indexed_knowledge) -> None:
    store, embedder = indexed_knowledge
    suite = load_eval_suite(write(tmp_path, VALID))

    payload = report_to_dict(run_eval(suite, embedder, store, k=5))
    round_tripped = json.loads(json.dumps(payload))

    assert round_tripped["suite_version"] == 1
    assert round_tripped["generated_at"].endswith("Z") or "T" in round_tripped["generated_at"]
    assert len(round_tripped["questions"]) == 2
    first = round_tripped["questions"][0]
    assert set(first) >= {"id", "text", "expected_projects", "found", "missing", "results"}
    assert set(first["results"][0]) >= {"rank", "project", "rel_path", "heading_path", "score"}
