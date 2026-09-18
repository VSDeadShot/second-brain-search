"""End-to-end indexing, offline: discover -> read -> chunk -> embed -> store."""

from __future__ import annotations

from pathlib import Path

import pytest

from second_brain.config import Config, DEFAULT_EXCLUDE_DIRS, DEFAULT_INCLUDE_PATTERNS
from second_brain.discovery import discover_documents
from second_brain.embedding import DailyQuotaExceeded, EmbeddingError, GeminiEmbedder
from second_brain.embedding_cache import EmbeddingCache
from second_brain.pipeline import (
    EmbeddingPlan,
    collect_chunks,
    embedding_plan,
    index_documents,
    store_chunks,
)
from second_brain.store import ChunkStore, VectorSpaceMismatch

from conftest import make_file, make_git_dir
from fakes import (
    FailingEmbedder,
    FakeClock,
    FakeEmbedder,
    StubClient,
    StubModels,
    daily_quota_error,
)


def config_for(root: Path) -> Config:
    return Config(
        scan_root=root,
        gemini_api_key=None,
        include_patterns=DEFAULT_INCLUDE_PATTERNS,
        include_dirs=("docs",),
        exclude_dirs=DEFAULT_EXCLUDE_DIRS,
        exclude_paths=(),
    )


def test_indexes_every_discovered_document(tmp_path: Path, corpus: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")

    report = index_documents(config_for(corpus), FakeEmbedder(), store)

    assert report.documents == 3
    assert report.projects == 2
    assert report.chunks > 0


def test_report_chunk_count_matches_the_store(tmp_path: Path, corpus: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")

    report = index_documents(config_for(corpus), FakeEmbedder(), store)

    assert store.count() == report.chunks


def test_reindexing_does_not_duplicate(tmp_path: Path, corpus: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")
    config = config_for(corpus)

    first = index_documents(config, FakeEmbedder(), store)
    second = index_documents(config, FakeEmbedder(), store)

    assert store.count() == first.chunks == second.chunks


def test_every_chunk_is_embedded_once(tmp_path: Path, corpus: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")
    embedder = FakeEmbedder()

    report = index_documents(config_for(corpus), embedder, store)

    embedded = sum(len(batch) for batch in embedder.embed_calls)
    assert embedded == report.chunks


def test_stored_chunks_carry_their_project(tmp_path: Path, corpus: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")

    index_documents(config_for(corpus), FakeEmbedder(), store)

    projects = {m["project"] for m in store.all_metadata()}
    assert projects == {"Alpha", "Beta"}


def test_empty_document_is_counted_but_yields_no_chunks(tmp_path: Path) -> None:
    root = tmp_path / "root"
    make_file(make_git_dir(root / "Solo") / "README.md", "   \n\n")
    store = ChunkStore(tmp_path / "chroma")

    report = index_documents(config_for(root), FakeEmbedder(), store)

    assert report.documents == 1
    assert report.chunks == 0


def test_undecodable_bytes_do_not_crash_the_run(tmp_path: Path) -> None:
    """Self-authored markdown should be UTF-8, but one bad file must not stop a rebuild."""
    root = tmp_path / "root"
    proj = make_git_dir(root / "Solo")
    (proj / "README.md").write_bytes(
        b"# Title\n\nvalid text \xff\xfe with invalid bytes, padded out far enough "
        b"that the chunk clears the size floor.\n"
    )
    store = ChunkStore(tmp_path / "chroma")

    report = index_documents(config_for(root), FakeEmbedder(), store)

    assert report.chunks > 0
    assert report.skipped == []


def test_file_deleted_after_discovery_is_skipped(tmp_path: Path, corpus: Path) -> None:
    """Discovery walks first, so a file can vanish before the pipeline reads it."""
    store = ChunkStore(tmp_path / "chroma")
    config = config_for(corpus)

    docs = discover_documents(config)
    (corpus / "Beta" / "README.md").unlink()

    report = index_documents(config, FakeEmbedder(), store, docs=docs)

    assert report.documents == 2
    assert len(report.skipped) == 1
    assert "Beta" in report.skipped[0][0]


def test_plain_upsert_leaves_stale_chunks_behind(tmp_path: Path, corpus: Path) -> None:
    """Why rebuild exists: without it, a deleted document stays searchable."""
    store = ChunkStore(tmp_path / "chroma")
    config = config_for(corpus)
    index_documents(config, FakeEmbedder(), store)

    (corpus / "Beta" / "README.md").unlink()
    index_documents(config, FakeEmbedder(), store)

    assert "Beta" in {m["project"] for m in store.all_metadata()}


def test_rebuild_removes_chunks_of_a_deleted_document(tmp_path: Path, corpus: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")
    config = config_for(corpus)
    index_documents(config, FakeEmbedder(), store, rebuild=True)

    (corpus / "Beta" / "README.md").unlink()
    report = index_documents(config, FakeEmbedder(), store, rebuild=True)

    assert {m["project"] for m in store.all_metadata()} == {"Alpha"}
    assert store.count() == report.chunks


def test_rebuild_removes_extra_chunks_of_a_shrunk_document(tmp_path: Path) -> None:
    """Deterministic ids mean chunks past the new end would otherwise survive."""
    root = tmp_path / "root"
    readme = make_git_dir(root / "Solo") / "README.md"
    section = "A section body long enough to become its own chunk in the index."
    make_file(readme, "\n\n".join(f"# Part {i}\n\n{section}" for i in range(5)))
    store = ChunkStore(tmp_path / "chroma")
    config = config_for(root)
    assert index_documents(config, FakeEmbedder(), store, rebuild=True).chunks == 5

    make_file(readme, f"# Part 0\n\n{section}")
    report = index_documents(config, FakeEmbedder(), store, rebuild=True)

    assert report.chunks == 1
    assert store.count() == 1


def test_failed_embed_during_rebuild_keeps_the_previous_index(
    tmp_path: Path, corpus: Path
) -> None:
    """Resetting before embedding would turn one API error into an empty index."""
    store = ChunkStore(tmp_path / "chroma")
    config = config_for(corpus)
    before = index_documents(config, FakeEmbedder(), store, rebuild=True)

    with pytest.raises(EmbeddingError):
        index_documents(config, FailingEmbedder(), store, rebuild=True)

    assert store.count() == before.chunks
    assert {m["project"] for m in store.all_metadata()} == {"Alpha", "Beta"}


def test_report_counts_chunks_per_project(tmp_path: Path, corpus: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")

    report = index_documents(config_for(corpus), FakeEmbedder(), store)

    assert set(report.chunks_per_project) == {"Alpha", "Beta"}
    assert sum(report.chunks_per_project.values()) == report.chunks


def test_store_chunks_forwards_batch_progress(tmp_path: Path, corpus: Path) -> None:
    seen: list[tuple[int, int]] = []

    store_chunks(
        collect_chunks(config_for(corpus)),
        FakeEmbedder(),
        ChunkStore(tmp_path / "chroma"),
        on_batch=lambda n, total: seen.append((n, total)),
    )

    assert seen == [(1, 1)]


# --- saved embeddings -----------------------------------------------------------


def paced_gemini(models: StubModels) -> GeminiEmbedder:
    """Real GeminiEmbedder over a stub client: 2 texts per batch, instant fake clock."""
    clock = FakeClock()
    models.clock = clock
    return GeminiEmbedder(
        client=StubClient(models=models),
        dimensions=models.dimensions,
        batch_size=2,
        items_per_minute=2,
        sleep=clock.sleep,
        clock=clock.monotonic,
    )


def embedded_text_count(models: StubModels) -> int:
    return sum(len(call["contents"]) for call in models.calls)


def test_embeddings_are_saved_as_they_are_produced(tmp_path: Path, corpus: Path) -> None:
    collected = collect_chunks(config_for(corpus))
    cache = EmbeddingCache(tmp_path / "cache.sqlite")

    store_chunks(collected, FakeEmbedder(), ChunkStore(tmp_path / "chroma"), cache=cache)

    assert cache.count() == len({c.content_hash for c in collected.chunks})


def test_second_run_reuses_every_saved_embedding(tmp_path: Path, corpus: Path) -> None:
    collected = collect_chunks(config_for(corpus))
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    store = ChunkStore(tmp_path / "chroma")
    store_chunks(collected, FakeEmbedder(), store, cache=cache, rebuild=True)

    second_embedder = FakeEmbedder()
    report = store_chunks(collected, second_embedder, store, cache=cache, rebuild=True)

    assert second_embedder.embed_calls == []
    assert (report.embedded, report.reused) == (0, len(collected.chunks))
    assert store.count() == len(collected.chunks)


def test_resumed_run_embeds_only_what_the_failed_run_missed(
    tmp_path: Path, corpus: Path
) -> None:
    """The live failure, in miniature: the daily cap hits on batch 2 of 2."""
    collected = collect_chunks(config_for(corpus))
    total = len(collected.chunks)
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    store = ChunkStore(tmp_path / "chroma")

    failing = StubModels(raise_on_calls={2: daily_quota_error()})
    with pytest.raises(DailyQuotaExceeded):
        store_chunks(collected, paced_gemini(failing), store, cache=cache, rebuild=True)
    assert cache.count() == 2  # batch 1 survived the failure
    assert store.count() == 0  # the index itself is still all-or-nothing

    resumed = StubModels()
    report = store_chunks(collected, paced_gemini(resumed), store, cache=cache, rebuild=True)

    assert embedded_text_count(resumed) == total - 2
    assert (report.embedded, report.reused) == (total - 2, 2)
    assert store.count() == total


def test_failed_run_with_saved_embeddings_still_keeps_the_previous_index(
    tmp_path: Path, corpus: Path
) -> None:
    collected = collect_chunks(config_for(corpus))
    store = ChunkStore(tmp_path / "chroma")
    store_chunks(collected, FakeEmbedder(), store, rebuild=True)
    before = store.count()

    failing = StubModels(raise_on_calls={2: daily_quota_error()})
    with pytest.raises(DailyQuotaExceeded):
        store_chunks(
            collected,
            paced_gemini(failing),
            store,
            cache=EmbeddingCache(tmp_path / "cache.sqlite"),
            rebuild=True,
        )

    assert store.count() == before


def test_changed_text_is_embedded_again(tmp_path: Path, corpus: Path) -> None:
    config = config_for(corpus)
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    store = ChunkStore(tmp_path / "chroma")
    store_chunks(collect_chunks(config), FakeEmbedder(), store, cache=cache, rebuild=True)

    make_file(
        corpus / "Beta" / "README.md",
        "# Beta\n\nBeta has changed completely, and now says something else entirely.",
    )
    report = store_chunks(collect_chunks(config), FakeEmbedder(), store, cache=cache, rebuild=True)

    assert (report.embedded, report.reused) == (1, 3)


def test_identical_texts_are_embedded_once(tmp_path: Path) -> None:
    """Every embedded text spends daily quota - two copies of a README should cost one."""
    root = tmp_path / "root"
    same = "# Shared\n\nThe same README text, copied verbatim into two projects."
    make_file(make_git_dir(root / "One") / "README.md", same)
    make_file(make_git_dir(root / "Two") / "README.md", same)
    collected = collect_chunks(config_for(root))
    embedder = FakeEmbedder()
    store = ChunkStore(tmp_path / "chroma")

    report = store_chunks(
        collected, embedder, store, cache=EmbeddingCache(tmp_path / "cache.sqlite")
    )

    assert embedder.embed_calls == [[same]]
    assert report.embedded == 1
    assert store.count() == 2


def test_saved_vectors_are_not_reused_across_vector_spaces(tmp_path: Path, corpus: Path) -> None:
    collected = collect_chunks(config_for(corpus))
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    store_chunks(collected, FakeEmbedder(dimensions=8), ChunkStore(tmp_path / "a"), cache=cache)

    report = store_chunks(
        collected, FakeEmbedder(dimensions=4), ChunkStore(tmp_path / "b"), cache=cache
    )

    assert report.reused == 0


def test_embedding_plan_counts_saved_and_pending(tmp_path: Path, corpus: Path) -> None:
    collected = collect_chunks(config_for(corpus))
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    embedder = FakeEmbedder()
    namespace = embedder.cache_namespace
    total = len(collected.chunks)

    assert embedding_plan(collected.chunks, namespace, cache) == EmbeddingPlan(reused=0, to_embed=total)

    store_chunks(collected, embedder, ChunkStore(tmp_path / "chroma"), cache=cache)

    assert embedding_plan(collected.chunks, namespace, cache) == EmbeddingPlan(reused=total, to_embed=0)


def test_embedding_plan_without_a_cache_counts_unique_texts(tmp_path: Path) -> None:
    root = tmp_path / "root"
    same = "# Shared\n\nThe same README text, copied verbatim into two projects."
    make_file(make_git_dir(root / "One") / "README.md", same)
    make_file(make_git_dir(root / "Two") / "README.md", same)
    collected = collect_chunks(config_for(root))

    assert embedding_plan(collected.chunks, "any", None) == EmbeddingPlan(reused=0, to_embed=1)


def test_without_a_cache_every_chunk_counts_as_embedded(tmp_path: Path, corpus: Path) -> None:
    collected = collect_chunks(config_for(corpus))

    report = store_chunks(collected, FakeEmbedder(), ChunkStore(tmp_path / "chroma"))

    assert (report.embedded, report.reused) == (len(collected.chunks), 0)


def test_embedder_receives_heading_prefixed_text_but_the_store_keeps_display_text(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    paragraph = "A paragraph about caching behaviour that runs on for a while. " * 3
    make_file(
        make_git_dir(root / "Solo") / "README.md",
        "# Architecture\n\n## Caching\n\n" + "\n\n".join([paragraph] * 12),
    )
    collected = collect_chunks(config_for(root))
    headless = next(c for c in collected.chunks if not c.text.lstrip().startswith("#"))
    embedder = FakeEmbedder()
    store = ChunkStore(tmp_path / "chroma")

    store_chunks(collected, embedder, store)

    sent = [text for call in embedder.embed_calls for text in call]
    assert headless.embed_text in sent
    assert headless.text not in sent
    assert store.get(headless.chunk_id)["text"] == headless.text


# --- vector space recorded in the index -----------------------------------------


def test_rebuild_records_the_embedders_vector_space(tmp_path: Path, corpus: Path) -> None:
    store = ChunkStore(tmp_path / "chroma")
    embedder = FakeEmbedder()

    store_chunks(collect_chunks(config_for(corpus)), embedder, store, rebuild=True)

    assert ChunkStore(tmp_path / "chroma").namespace == embedder.cache_namespace


def test_first_write_to_a_fresh_store_records_the_vector_space(
    tmp_path: Path, corpus: Path
) -> None:
    store = ChunkStore(tmp_path / "chroma")
    embedder = FakeEmbedder()

    store_chunks(collect_chunks(config_for(corpus)), embedder, store)

    assert store.namespace == embedder.cache_namespace


def test_mixing_vector_spaces_in_one_index_is_refused_before_embedding(
    tmp_path: Path, corpus: Path
) -> None:
    """Same dimensions, different model: Chroma would accept the vectors silently."""
    collected = collect_chunks(config_for(corpus))
    store = ChunkStore(tmp_path / "chroma")
    store_chunks(collected, FakeEmbedder(), store)
    other = FailingEmbedder()  # same 8 dims, different namespace

    with pytest.raises(VectorSpaceMismatch):
        store_chunks(collected, other, store)

    assert other.embed_calls == []


def test_rebuild_may_switch_vector_space(tmp_path: Path, corpus: Path) -> None:
    """A rebuild replaces the whole index, so changing embedders is legitimate there."""
    collected = collect_chunks(config_for(corpus))
    store = ChunkStore(tmp_path / "chroma")
    store_chunks(collected, FakeEmbedder(dimensions=8), store, rebuild=True)

    store_chunks(collected, FakeEmbedder(dimensions=4), store, rebuild=True)

    assert store.namespace == FakeEmbedder(dimensions=4).cache_namespace


def test_collect_chunks_matches_what_indexing_stores(tmp_path: Path, corpus: Path) -> None:
    """The dry run relies on this: same chunks, no embedder, no store."""
    config = config_for(corpus)
    store = ChunkStore(tmp_path / "chroma")

    collected = collect_chunks(config)
    report = index_documents(config, FakeEmbedder(), store)

    assert len(collected.chunks) == report.chunks
    assert collected.report.chunks_per_project == report.chunks_per_project
    assert {c.chunk_id for c in collected.chunks} == {m for m in store.all_ids()}
