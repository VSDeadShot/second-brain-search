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
COMMITTED_SUITE = REPO_ROOT / "eval" / "retrieval_v2.toml"

VALID = """
version = 2

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
    body = '\nversion = 2\n\n[[question]]\nid = "q1"\ncategory = "x"\n'

    with pytest.raises(EvalSuiteError, match="q1"):
        load_eval_suite(write(tmp_path, body))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    body = VALID + '\n[[question]]\nid = "q1"\ncategory = "x"\ntext = "t"\nexpected_projects = []\n'

    with pytest.raises(EvalSuiteError, match="q1"):
        load_eval_suite(write(tmp_path, body))


def test_an_unknown_version_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvalSuiteError, match="version"):
        load_eval_suite(write(tmp_path, VALID.replace("version = 2", "version = 99")))


def test_version_1_is_refused(tmp_path: Path) -> None:
    """v2 changed the questions and the scoring. A v1 file scored by v2 rules would
    produce a report that looks comparable with old v1 runs and isn't."""
    with pytest.raises(EvalSuiteError, match="version"):
        load_eval_suite(write(tmp_path, VALID.replace("version = 2", "version = 1")))


def test_a_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(EvalSuiteError, match="not found"):
        load_eval_suite(tmp_path / "nope.toml")


def test_the_committed_suite_splits_q1_into_a_fact_and_a_rationale() -> None:
    suite = load_eval_suite(COMMITTED_SUITE)
    by_id = {q.id: q for q in suite.questions}

    assert suite.version == 2
    assert len(suite.questions) == len(by_id) == 11
    assert all(q.text and q.category for q in suite.questions)
    assert "q1-flashcards-model" not in by_id
    # Q1a scores what the corpus states. It's stale - the owner runs 3.6 - but it
    # is the fact retrieval can find, so it is the one retrieval is scored on.
    assert by_id["q1a-flashcards-model-default"].expected_text == ("gemini-3.7-flash",)
    # Nothing in the corpus answers these three.
    assert by_id["q1b-flashcards-model-why"].check == "no-answer"
    assert by_id["q10-kubernetes"].check == "no-answer"
    assert by_id["q5-dsa-review-security"].check == "no-answer"
    # Q5's fix lives only in git history, which the index does not cover.
    assert "git" in by_id["q5-dsa-review-security"].notes.lower()


def test_fact_questions_carry_the_string_that_states_the_fact() -> None:
    by_id = {q.id: q for q in load_eval_suite(COMMITTED_SUITE).questions}

    # Two verbatim wordings of one fact: CLAUDE.md's and api-contract.md's.
    assert by_id["q2-generation-limit"].expected_text == (
        "20 calls per owner per day",
        "20 generations per owner per day",
    )
    assert by_id["q4-omnitask-lost-writes"].expected_text == ("swaps the file in atomically",)
    assert by_id["q9-signin-hammering"].expected_text == ("LoginRateLimit",)
    # Q3 has no string rare enough to trust, and the cross-project questions are
    # answered by the projects themselves: all stay on the project check.
    for qid in ("q3-watch-next-auth", "q6-rate-limiting", "q7-gemini-projects", "q8-spaced-repetition"):
        assert by_id[qid].check == "project"


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


def _question(*, expected: tuple[str, ...], texts: tuple[str, ...] = ()):
    from second_brain.evaluation import EvalQuestion

    return EvalQuestion(
        id="qx",
        category="c",
        text="t",
        expected_projects=expected,
        also_in_corpus=(),
        notes="",
        expected_text=texts,
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

    assert round_tripped["suite_version"] == 2
    assert round_tripped["generated_at"].endswith("Z") or "T" in round_tripped["generated_at"]
    assert len(round_tripped["questions"]) == 2
    first = round_tripped["questions"][0]
    assert set(first) >= {"id", "text", "expected_projects", "found", "missing", "results"}
    assert set(first["results"][0]) >= {"rank", "project", "rel_path", "heading_path", "score"}


# --- fact-level checks: expected_text --------------------------------------------
#
# Q1 was refused by `sbs ask` because the model name never reached the passages,
# yet it scored a pass: the right project came back, which was all the project
# check could see. expected_text asks the question that matters - did a passage
# carrying the fact come back?

FACT = """
version = 2

[[question]]
id = "q1a"
category = "exact-fact"
text = "Which Gemini model does Interview Flashcards use by default?"
expected_projects = ["Interview Flashcards"]
expected_text = ["gemini-3.7-flash", "gemini 3.7 flash"]
"""

MODEL = ("gemini-3.7-flash",)


def passage(
    text: str,
    *,
    project: str = "Interview Flashcards",
    rel_path: str = "plan.md",
    heading: str = "H",
    index: int = 0,
    score: float = 0.7,
) -> RetrievedChunk:
    return RetrievedChunk(
        project=project,
        rel_path=rel_path,
        heading_path=heading,
        chunk_index=index,
        text=text,
        score=score,
        content_hash=f"h{index}",
        mtime=1.0,
        path=f"/{project}/{rel_path}",
    )


def test_expected_text_is_loaded_as_a_tuple(tmp_path: Path) -> None:
    question = load_eval_suite(write(tmp_path, FACT)).questions[0]

    assert question.expected_text == ("gemini-3.7-flash", "gemini 3.7 flash")


@pytest.mark.parametrize("value", ['"gemini-3.7-flash"', "[]", '[""]', '["   "]', "[3]"])
def test_expected_text_must_be_a_list_of_non_blank_strings(tmp_path: Path, value: str) -> None:
    body = FACT.replace('["gemini-3.7-flash", "gemini 3.7 flash"]', value)

    with pytest.raises(EvalSuiteError, match="expected_text must be"):
        load_eval_suite(write(tmp_path, body))


def test_expected_text_on_a_no_answer_question_is_rejected(tmp_path: Path) -> None:
    """"Expects nothing" and "expects this text" can't both be true."""
    body = FACT.replace('expected_projects = ["Interview Flashcards"]', "expected_projects = []")

    with pytest.raises(EvalSuiteError, match="no answer"):
        load_eval_suite(write(tmp_path, body))


def test_questions_without_expected_text_load_as_before(tmp_path: Path) -> None:
    """Guard, green from the start: the field is optional."""
    assert load_eval_suite(write(tmp_path, VALID)).questions[0].expected_text == ()


@pytest.mark.parametrize(
    ("projects", "texts", "check"),
    [((), (), "no-answer"), (("A",), (), "project"), (("A",), ("x",), "text")],
)
def test_the_check_follows_the_fields_present(projects, texts, check) -> None:
    assert _question(expected=projects, texts=texts).check == check


def test_the_right_project_without_the_fact_fails() -> None:
    """Q1's bug as a test. Every passage is from the right project, none carries
    the fact - and the project check alone calls that a pass."""
    outcome = evaluate_question(
        _question(expected=("Interview Flashcards",), texts=MODEL),
        [passage("Generation is a capability, not a precondition."), passage("The key is optional.", index=1)],
    )

    assert outcome.found == ("Interview Flashcards",)
    assert outcome.passed is False


def test_a_passage_containing_the_string_passes() -> None:
    outcome = evaluate_question(
        _question(expected=("Interview Flashcards",), texts=MODEL),
        [passage("intro"), passage('model = "gemini-3.7-flash";', index=21)],
    )

    assert outcome.passed is True
    assert [m.text for m in outcome.text_matches] == ["gemini-3.7-flash"]


def test_matching_ignores_case() -> None:
    outcome = evaluate_question(
        _question(expected=("Interview Flashcards",), texts=MODEL), [passage("Default: Gemini-3.7-Flash.")]
    )

    assert outcome.passed is True
    assert [m.text for m in outcome.text_matches] == ["gemini-3.7-flash"]


def test_a_string_broken_by_a_line_wrap_still_matches() -> None:
    """Every doc in the corpus is hard-wrapped. api-contract.md splits this exact
    phrase across two lines; a strict substring could never find it."""
    outcome = evaluate_question(
        _question(expected=("Interview Flashcards",), texts=("20 generations per owner per day",)),
        [passage("rationed: **20 generations per\nowner per day**, counted from")],
    )

    assert outcome.passed is True
    assert [m.text for m in outcome.text_matches] == ["20 generations per owner per day"]


def test_any_one_of_several_strings_is_enough() -> None:
    outcome = evaluate_question(
        _question(expected=("Interview Flashcards",), texts=("gemini-3.7-flash", "LoginRateLimit")),
        [passage("`LoginRateLimit` is its own bean.")],
    )

    assert outcome.passed is True
    assert [m.text for m in outcome.text_matches] == ["LoginRateLimit"]


def test_the_match_is_a_substring_not_a_fuzzy_one() -> None:
    """Near misses don't count: a prefix, or spaces where the string has hyphens."""
    outcome = evaluate_question(
        _question(expected=("Interview Flashcards",), texts=MODEL),
        [passage("upgraded from gemini-3.7 last week"), passage("gemini 3.7 flash", index=1)],
    )

    assert outcome.passed is False


def test_a_string_only_in_the_heading_does_not_count() -> None:
    """A heading that names the fact is not a passage stating it."""
    outcome = evaluate_question(
        _question(expected=("Interview Flashcards",), texts=MODEL),
        [passage("Unrelated body text.", heading="Default model: gemini-3.7-flash")],
    )

    assert outcome.passed is False


def test_a_match_reports_its_rank_and_location() -> None:
    from second_brain.evaluation import TextMatch

    outcome = evaluate_question(
        _question(expected=("Interview Flashcards",), texts=MODEL),
        [
            passage("a", index=3),
            passage('model = "gemini-3.7-flash";', rel_path="plans/p.md", index=21),
            passage("gemini-3.7-flash again", rel_path="plans/p.md", index=23),
        ],
    )

    assert outcome.text_matches == (
        TextMatch("gemini-3.7-flash", 2, "Interview Flashcards", "plans/p.md", 21),
    )


def test_a_no_answer_question_is_neither_passed_nor_failed() -> None:
    outcome = evaluate_question(_question(expected=()), [chunk("B", score=0.64)])

    assert outcome.passed is None


def test_the_project_check_is_unchanged_without_expected_text() -> None:
    """Guard, green from the start: questions without expected_text score as before."""
    both = evaluate_question(_question(expected=("A", "B")), [chunk("A"), chunk("B")])
    one = evaluate_question(_question(expected=("A", "B")), [chunk("A")])

    assert both.passed is True
    assert one.passed is False


def test_every_outcome_says_why() -> None:
    hit = evaluate_question(
        _question(expected=("Interview Flashcards",), texts=MODEL),
        [passage("x"), passage("gemini-3.7-flash", index=21)],
    )
    miss = evaluate_question(_question(expected=("Interview Flashcards",), texts=MODEL), [passage("x")])
    project = evaluate_question(_question(expected=("A",)), [chunk("B")])
    none = evaluate_question(_question(expected=()), [chunk("B", score=0.64)])

    assert "gemini-3.7-flash" in hit.reason and "rank 2" in hit.reason
    assert "none of" in miss.reason and "gemini-3.7-flash" in miss.reason
    assert "Interview Flashcards" in miss.reason  # the project DID come back - say so
    assert "missing A" in project.reason
    assert "top score 0.640" in none.reason


# --- the summary and the report carry the verdicts --------------------------------

VERDICTS = """
version = 2

[[question]]
id = "fact-hit"
category = "exact-fact"
text = "caching redis ttl"
expected_projects = ["Watch Tracker"]
expected_text = ["show metadata in redis"]

[[question]]
id = "fact-miss"
category = "exact-fact"
text = "caching redis ttl"
expected_projects = ["Watch Tracker"]
expected_text = ["memcached"]

[[question]]
id = "project"
category = "cross-project"
text = "write ahead log storage"
expected_projects = ["RDBMS"]

[[question]]
id = "none"
category = "trap"
text = "kubernetes helm pods"
expected_projects = []
"""


def test_summary_counts_passes_per_check(tmp_path: Path, indexed_knowledge) -> None:
    store, embedder = indexed_knowledge
    suite = load_eval_suite(write(tmp_path, VERDICTS))

    summary = run_eval(suite, embedder, store, k=5).summary

    assert (summary["passed"], summary["failed"], summary["no_answer"]) == (2, 1, 1)
    assert summary["by_check"] == {
        "text": {"scored": 2, "passed": 1},
        "project": {"scored": 1, "passed": 1},
    }
    # The old project-level counts stay, so v2 reports can be read beside v1 ones.
    assert summary["all_expected_found"] == 3


def test_report_json_carries_the_check_and_reason(tmp_path: Path, indexed_knowledge) -> None:
    store, embedder = indexed_knowledge
    suite = load_eval_suite(write(tmp_path, VERDICTS))

    questions = {q["id"]: q for q in report_to_dict(run_eval(suite, embedder, store, k=5))["questions"]}

    hit = questions["fact-hit"]
    assert (hit["check"], hit["passed"]) == ("text", True)
    assert hit["expected_text"] == ["show metadata in redis"]
    assert set(hit["text_matches"][0]) == {"text", "rank", "project", "rel_path", "chunk_index"}
    assert hit["reason"]
    assert (questions["fact-miss"]["check"], questions["fact-miss"]["passed"]) == ("text", False)
    assert (questions["none"]["check"], questions["none"]["passed"]) == ("no-answer", None)
    json.dumps(questions)  # still serialisable


def test_report_results_carry_the_chunk_index(tmp_path: Path, indexed_knowledge) -> None:
    """Passages from one section share a heading path; only the chunk index tells
    them apart. Without it, Q2's report couldn't say which Errors chunk came back."""
    store, embedder = indexed_knowledge
    suite = load_eval_suite(write(tmp_path, VERDICTS))

    questions = report_to_dict(run_eval(suite, embedder, store, k=5))["questions"]

    for result in questions[0]["results"]:
        assert isinstance(result["chunk_index"], int)
