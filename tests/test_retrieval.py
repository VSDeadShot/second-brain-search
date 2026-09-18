"""Retrieval: embed a query, search the index, return ranked cited chunks.

Runs against a real Chroma store in tmp_path, with KeywordEmbedder standing in
for Gemini. These tests prove the plumbing - ranking by vector similarity,
limits, filters, dedup, refusals. They cannot prove that Gemini's embeddings
rank the right chunks for real questions; that needs the real index.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from second_brain.config import Config, DEFAULT_EXCLUDE_DIRS, DEFAULT_INCLUDE_PATTERNS
from second_brain.embedding import DailyQuotaExceeded
from second_brain.pipeline import collect_chunks, store_chunks
from second_brain.retrieval import RetrievalError, RetrievedChunk, retrieve
from second_brain.store import ChunkStore

from conftest import make_file, make_git_dir
from fakes import FakeEmbedder, KeywordEmbedder


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


# --- ranking --------------------------------------------------------------------


def test_most_relevant_chunk_ranks_first(indexed) -> None:
    store, embedder = indexed

    results = retrieve("how did I handle photo compression", embedder, store, k=3)

    assert results[0].project == "Macro Tracker"


def test_each_topic_finds_its_own_project(indexed) -> None:
    store, embedder = indexed

    tops = {
        query: retrieve(query, embedder, store, k=1)[0].project
        for query in ("caching with redis", "write ahead log storage", "jpeg compression")
    }

    assert tops == {
        "caching with redis": "Watch Tracker",
        "write ahead log storage": "RDBMS",
        "jpeg compression": "Macro Tracker",
    }


def test_scores_descend_and_are_cosine_similarities(indexed) -> None:
    store, embedder = indexed

    scores = [r.score for r in retrieve("caching photo storage", embedder, store, k=3)]

    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 - 1e-6 <= s <= 1.0 + 1e-6 for s in scores)


def test_k_limits_the_number_of_results(indexed) -> None:
    store, embedder = indexed

    assert len(retrieve("caching", embedder, store, k=2)) == 2


def test_k_larger_than_the_index_returns_everything(indexed) -> None:
    store, embedder = indexed

    assert len(retrieve("caching", embedder, store, k=50)) == 3


def test_k_must_be_positive(indexed) -> None:
    store, embedder = indexed

    with pytest.raises(ValueError):
        retrieve("caching", embedder, store, k=0)


# --- results carry what a citation needs ----------------------------------------


def test_results_carry_the_full_citation(indexed) -> None:
    store, embedder = indexed

    top = retrieve("redis caching ttl", embedder, store, k=1)[0]

    assert isinstance(top, RetrievedChunk)
    assert (top.project, top.rel_path, top.heading_path) == ("Watch Tracker", "EXPLAINER.md", "Caching")
    assert top.chunk_index == 0
    assert "redis" in top.text
    assert top.content_hash


def test_results_show_display_text_not_the_embedded_prefix(tmp_path: Path) -> None:
    root = tmp_path / "root"
    paragraph = "Caching details that run on for a good while in this section. " * 3
    make_file(
        make_git_dir(root / "Solo") / "README.md",
        "# Architecture\n\n## Caching\n\n" + "\n\n".join([paragraph] * 12),
    )
    store, embedder = ChunkStore(tmp_path / "chroma"), KeywordEmbedder()
    store_chunks(collect_chunks(config_for(root)), embedder, store, rebuild=True)

    results = retrieve("caching details", embedder, store, k=10)

    assert not any(r.text.startswith("Architecture > Caching") for r in results)


# --- project filter -------------------------------------------------------------


def test_project_filter_limits_results_to_that_project(indexed) -> None:
    store, embedder = indexed

    results = retrieve("photo compression", embedder, store, k=5, project="RDBMS")

    assert [r.project for r in results] == ["RDBMS"]


def test_project_name_matches_case_insensitively(indexed) -> None:
    store, embedder = indexed

    results = retrieve("caching", embedder, store, k=5, project="watch tracker")

    assert {r.project for r in results} == {"Watch Tracker"}


def test_unknown_project_lists_valid_names_without_embedding(indexed) -> None:
    store, embedder = indexed

    with pytest.raises(RetrievalError) as exc_info:
        retrieve("caching", embedder, store, project="Nope")

    message = str(exc_info.value)
    assert "Nope" in message
    for name in ("Macro Tracker", "RDBMS", "Watch Tracker"):
        assert name in message
    assert embedder.query_calls == []


# --- duplicates -----------------------------------------------------------------


def test_identical_texts_collapse_to_one_result(tmp_path: Path, knowledge: Path) -> None:
    """The real corpus has 11 duplicate-text chunks; two copies shouldn't fill two slots."""
    make_file(
        make_git_dir(knowledge / "Watch Tracker Copy") / "EXPLAINER.md",
        (knowledge / "Watch Tracker" / "EXPLAINER.md").read_text(encoding="utf-8"),
    )
    store, embedder = ChunkStore(tmp_path / "chroma"), KeywordEmbedder()
    store_chunks(collect_chunks(config_for(knowledge)), embedder, store, rebuild=True)

    results = retrieve("redis caching ttl", embedder, store, k=2)

    assert len({r.content_hash for r in results}) == len(results) == 2
    assert sum(1 for r in results if "redis" in r.text) == 1


# --- refusals: all before any query is embedded (no quota spent) ----------------


def test_blank_query_is_refused_without_embedding(indexed) -> None:
    store, embedder = indexed

    with pytest.raises(RetrievalError, match="empty"):
        retrieve("   ", embedder, store)

    assert embedder.query_calls == []


def test_empty_index_is_refused_without_embedding(tmp_path: Path) -> None:
    embedder = KeywordEmbedder()

    with pytest.raises(RetrievalError, match="sbs index"):
        retrieve("caching", embedder, ChunkStore(tmp_path / "chroma"))

    assert embedder.query_calls == []


def test_index_built_in_another_vector_space_is_refused_without_embedding(
    tmp_path: Path, knowledge: Path
) -> None:
    store = ChunkStore(tmp_path / "chroma")
    store_chunks(collect_chunks(config_for(knowledge)), FakeEmbedder(dimensions=256), store, rebuild=True)
    embedder = KeywordEmbedder(dimensions=256)  # same dimensions, different space

    with pytest.raises(RetrievalError) as exc_info:
        retrieve("caching", embedder, store)

    assert "fake|256" in str(exc_info.value)
    assert "keyword|256" in str(exc_info.value)
    assert embedder.query_calls == []


def test_index_with_no_recorded_vector_space_is_refused(tmp_path: Path, knowledge: Path) -> None:
    """An index written before spaces were recorded can't be trusted - rebuild it."""
    store = ChunkStore(tmp_path / "chroma")
    chunks = collect_chunks(config_for(knowledge)).chunks
    store.upsert(chunks, KeywordEmbedder().embed_documents([c.embed_text for c in chunks]))

    with pytest.raises(RetrievalError, match="sbs index"):
        retrieve("caching", KeywordEmbedder(), store)


# --- the query itself -----------------------------------------------------------


def test_the_query_is_embedded_as_a_query(indexed) -> None:
    """Gemini uses a different task type for queries than for documents."""
    store, embedder = indexed
    embedder.embed_calls.clear()

    retrieve("caching", embedder, store)

    assert embedder.query_calls == ["caching"]
    assert embedder.embed_calls == []


def test_quota_errors_while_embedding_the_query_propagate(indexed) -> None:
    store, _ = indexed

    class QuotaSpentEmbedder(KeywordEmbedder):
        def embed_query(self, text: str) -> list[float]:
            raise DailyQuotaExceeded("daily limit of 1000 embedded texts is used up")

    with pytest.raises(DailyQuotaExceeded):
        retrieve("caching", QuotaSpentEmbedder(), store)
