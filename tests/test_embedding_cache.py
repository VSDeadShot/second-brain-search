"""Saved embeddings, keyed by what produced them and the exact text embedded."""

from __future__ import annotations

from pathlib import Path

from second_brain.embedding_cache import EmbeddingCache, cache_key


def test_vectors_round_trip_exactly(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    vector = [0.1, -0.25, 1 / 3, 0.0]

    cache.put_many([("k1", vector)])

    assert cache.get_many(["k1"]) == {"k1": vector}


def test_unknown_keys_are_simply_absent(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    cache.put_many([("k1", [1.0])])

    assert cache.get_many(["k1", "missing"]) == {"k1": [1.0]}


def test_saved_vectors_survive_a_new_instance(tmp_path: Path) -> None:
    """The whole point: a failed run's work is still there next time."""
    path = tmp_path / "cache.sqlite"
    EmbeddingCache(path).put_many([("k1", [0.5, 0.5])])

    assert EmbeddingCache(path).get_many(["k1"]) == {"k1": [0.5, 0.5]}


def test_putting_a_key_again_replaces_it(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    cache.put_many([("k1", [1.0, 0.0])])

    cache.put_many([("k1", [0.0, 1.0])])

    assert cache.get_many(["k1"]) == {"k1": [0.0, 1.0]}
    assert cache.count() == 1


def test_large_lookups_are_not_limited_by_sqlite_parameter_caps(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    items = [(f"k{i}", [float(i)]) for i in range(1200)]
    cache.put_many(items)

    found = cache.get_many([k for k, _ in items])

    assert len(found) == 1200


def test_different_vector_sizes_coexist(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.sqlite")

    cache.put_many([("small", [1.0] * 4), ("large", [1.0] * 768)])

    assert len(cache.get_many(["large"])["large"]) == 768


def test_creates_its_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "data" / "cache.sqlite"

    EmbeddingCache(path).put_many([("k1", [1.0])])

    assert path.exists()


def test_open_existing_does_not_create_a_missing_cache(tmp_path: Path) -> None:
    """A dry run reads the cache; it must never create one."""
    path = tmp_path / "data" / "cache.sqlite"

    assert EmbeddingCache.open_existing(path) is None
    assert not path.exists()
    assert not path.parent.exists()


def test_open_existing_reads_a_cache_that_is_there(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    EmbeddingCache(path).put_many([("k1", [1.0])])

    cache = EmbeddingCache.open_existing(path)

    assert cache is not None
    assert cache.get_many(["k1"]) == {"k1": [1.0]}


def test_key_separates_namespaces_for_the_same_text() -> None:
    """Changing model or dimensions must never reuse an incompatible vector."""
    assert cache_key("model-a|768", "abc") != cache_key("model-a|3072", "abc")
    assert cache_key("model-a|768", "abc") == cache_key("model-a|768", "abc")
