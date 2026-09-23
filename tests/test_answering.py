"""Answering: what the model said, and whether it may be shown.

Offline throughout - a scripted FakeGenerator stands in for Gemini, and a real
Chroma store with KeywordEmbedder supplies the passages. What is under test is
the policy: an answer nobody can trace to a passage is not an answer.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from second_brain.answering import Answer, Citation, answer_question
from second_brain.config import Config, DEFAULT_EXCLUDE_DIRS, DEFAULT_INCLUDE_PATTERNS
from second_brain.freshness import DateLookup
from second_brain.generation import GenerationError, RawAnswer
from second_brain.pipeline import collect_chunks, store_chunks
from second_brain.retrieval import RetrievalError
from second_brain.store import ChunkStore

from fakes import FakeGenerator, FakeRunner, KeywordEmbedder


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
def indexed(tmp_path: Path, knowledge: Path) -> tuple[ChunkStore, KeywordEmbedder]:
    store = ChunkStore(tmp_path / "chroma")
    embedder = KeywordEmbedder()
    store_chunks(collect_chunks(config_for(knowledge)), embedder, store, rebuild=True)
    return store, embedder


@pytest.fixture
def dates() -> DateLookup:
    return DateLookup(runner=FakeRunner("2026-08-18\n"))


def ask(indexed, generator: FakeGenerator, dates: DateLookup, question: str = "caching redis ttl", **kwargs) -> Answer:
    store, embedder = indexed
    return answer_question(question, embedder, store, generator, dates=dates, **kwargs)


# --- an answer that cites its sources -------------------------------------------


def test_a_cited_answer_is_shown(indexed, dates) -> None:
    generator = FakeGenerator(RawAnswer(True, "Redis, with a ttl [1].", (1,)))

    answer = ask(indexed, generator, dates)

    assert answer.answerable
    assert answer.text == "Redis, with a ttl [1]."
    assert answer.warnings == ()


def test_a_citation_points_at_the_passage_the_number_names(indexed, dates) -> None:
    generator = FakeGenerator(RawAnswer(True, "See [2].", (2,)))

    answer = ask(indexed, generator, dates)

    assert [c.number for c in answer.citations] == [2]
    assert answer.citations[0].chunk == answer.passages[1]


def test_the_model_that_answered_is_recorded(indexed, dates) -> None:
    generator = FakeGenerator(model="gemini-3.6-flash")

    assert ask(indexed, generator, dates).model == "gemini-3.6-flash"


def test_the_question_is_carried_on_the_answer(indexed, dates) -> None:
    assert ask(indexed, FakeGenerator(), dates, question="what about redis").question == "what about redis"


# --- citations that do not hold up ----------------------------------------------


def test_a_citation_number_with_no_such_passage_is_dropped(indexed, dates) -> None:
    generator = FakeGenerator(RawAnswer(True, "See [1] and [99].", (1, 99)))

    answer = ask(indexed, generator, dates)

    assert [c.number for c in answer.citations] == [1]
    assert any("99" in w for w in answer.warnings)


def test_a_citation_number_of_zero_or_less_is_dropped(indexed, dates) -> None:
    generator = FakeGenerator(RawAnswer(True, "See [1].", (0, -3, 1)))

    answer = ask(indexed, generator, dates)

    assert [c.number for c in answer.citations] == [1]


def test_the_same_citation_twice_is_listed_once(indexed, dates) -> None:
    generator = FakeGenerator(RawAnswer(True, "See [1] and again [1].", (1, 1)))

    assert [c.number for c in ask(indexed, generator, dates).citations] == [1]


def test_an_answer_citing_nothing_is_not_shown_as_an_answer(indexed, dates) -> None:
    """The whole point of the tool is traceability. An uncited claim is a guess,
    however confident it sounds."""
    generator = FakeGenerator(RawAnswer(True, "It uses redis, obviously.", ()))

    answer = ask(indexed, generator, dates)

    assert not answer.answerable
    assert "It uses redis, obviously." not in answer.text
    assert any("cited no" in w.lower() for w in answer.warnings)


def test_an_answer_whose_every_citation_is_invalid_is_not_shown(indexed, dates) -> None:
    generator = FakeGenerator(RawAnswer(True, "It uses redis [42].", (42,)))

    answer = ask(indexed, generator, dates)

    assert not answer.answerable
    assert "redis [42]" not in answer.text


def test_an_answerable_response_with_no_text_is_a_refusal(indexed, dates) -> None:
    generator = FakeGenerator(RawAnswer(True, "   ", (1,)))

    assert not ask(indexed, generator, dates).answerable


# --- refusals -------------------------------------------------------------------


def test_a_refusal_carries_no_citations_but_keeps_the_passages(indexed, dates) -> None:
    """The CLI shows the closest passages, labelled as not an answer."""
    generator = FakeGenerator(RawAnswer(False, "", ()))

    answer = ask(indexed, generator, dates, question="how did I set up kubernetes")

    assert not answer.answerable
    assert answer.citations == ()
    assert len(answer.passages) >= 3


def test_a_refusal_that_still_sent_citations_shows_none(indexed, dates) -> None:
    generator = FakeGenerator(RawAnswer(False, "", (1, 2)))

    assert ask(indexed, generator, dates).citations == ()


def test_nothing_retrieved_means_nothing_is_generated(indexed, dates, monkeypatch) -> None:
    """Defensive: generating from no passages spends quota to be told nothing."""
    monkeypatch.setattr("second_brain.answering.retrieve_for_answer", lambda *a, **k: [])
    store, embedder = indexed
    generator = FakeGenerator()

    answer = answer_question("caching", embedder, store, generator, dates=dates)

    assert not answer.answerable
    assert answer.passages == ()
    assert generator.prompts == []


# --- dates on citations ---------------------------------------------------------


def test_a_citation_carries_the_date_git_gives(indexed) -> None:
    generator = FakeGenerator(RawAnswer(True, "See [1].", (1,)))

    answer = ask(indexed, generator, DateLookup(runner=FakeRunner("2026-08-18\n")))

    assert answer.citations[0].date == "2026-08-18"
    assert answer.citations[0].date_source == "git"


def test_a_citation_falls_back_to_the_files_own_date(indexed) -> None:
    generator = FakeGenerator(RawAnswer(True, "See [1].", (1,)))

    answer = ask(indexed, generator, DateLookup(runner=FakeRunner(raises=FileNotFoundError("git"))))

    assert answer.citations[0].date_source == "mtime"


def test_a_file_edited_since_indexing_is_flagged_on_its_citation(indexed, dates) -> None:
    generator = FakeGenerator(RawAnswer(True, "See [1].", (1,)))
    answer = ask(indexed, generator, dates)
    edited = Path(answer.citations[0].chunk.path)
    edited.write_text(edited.read_text(encoding="utf-8") + "\n\nNew section.\n", encoding="utf-8")
    # Stamped an hour on, so the test doesn't hinge on how long it took to run:
    # a real edit is minutes or days after indexing, never milliseconds.
    later = edited.stat().st_mtime + 3600
    os.utime(edited, (later, later))

    after = ask(indexed, FakeGenerator(RawAnswer(True, "See [1].", (1,))), dates)

    assert after.citations[0].changed_since_indexed


def test_an_unchanged_file_is_not_flagged(indexed, dates) -> None:
    generator = FakeGenerator(RawAnswer(True, "See [1].", (1,)))

    assert not ask(indexed, generator, dates).citations[0].changed_since_indexed


# --- what the generator is given ------------------------------------------------


def test_the_generator_sees_exactly_the_retrieved_passages(indexed, dates) -> None:
    generator = FakeGenerator()

    answer = ask(indexed, generator, dates)

    assert generator.passages[0] == list(answer.passages)


def test_k_and_project_reach_retrieval(indexed, dates) -> None:
    generator = FakeGenerator()

    answer = ask(indexed, generator, dates, k=2, project="Watch Tracker")

    assert len(answer.passages) <= 2
    assert {p.project for p in answer.passages} == {"Watch Tracker"}


def test_a_generation_failure_is_not_swallowed(indexed, dates) -> None:
    generator = FakeGenerator(raises=GenerationError("Gemini is rate limiting"))

    with pytest.raises(GenerationError):
        ask(indexed, generator, dates)


def test_an_empty_question_never_reaches_the_generator(indexed, dates) -> None:
    generator = FakeGenerator()

    with pytest.raises(RetrievalError):
        ask(indexed, generator, dates, question="   ")

    assert generator.prompts == []


def test_citations_are_ordered_by_passage_number(indexed, dates) -> None:
    generator = FakeGenerator(RawAnswer(True, "See [3] and [1].", (3, 1)))

    assert [c.number for c in ask(indexed, generator, dates).citations] == [1, 3]


def test_a_citation_is_a_citation_and_an_answer_is_an_answer(indexed, dates) -> None:
    answer = ask(indexed, FakeGenerator(), dates)

    assert isinstance(answer, Answer)
    assert all(isinstance(c, Citation) for c in answer.citations)
