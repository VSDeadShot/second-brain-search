"""Query plumbing shared by the CLI and the MCP server: the index location, the
no-index guard, the Gemini embedder factory, and the fail-closed list of projects
the MCP server hides. No network: the SDK client is never built."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from second_brain import cli
from second_brain.chunking import chunk_document
from second_brain.config import (
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_INCLUDE_PATTERNS,
    Config,
    ConfigError,
)
from second_brain.embedding import EmbeddingError, GeminiEmbedder
from second_brain.pipeline import collect_chunks, store_chunks
from second_brain.runtime import (
    DEFAULT_INDEX_DIR,
    excluded_projects,
    existing_chunk_count,
    gemini_embedder,
)
from second_brain.store import ChunkStore

from fakes import FakeEmbedder, KeywordEmbedder

REPO_ROOT = Path(__file__).resolve().parents[1]

REALISTIC = (
    "# Title\n\nThis body is long enough to clear the chunking size floor, "
    "the way any real project document would be."
)


def config_for(root: Path, api_key: str | None) -> Config:
    return Config(
        scan_root=root,
        gemini_api_key=api_key,
        include_patterns=DEFAULT_INCLUDE_PATTERNS,
        include_dirs=("docs",),
        exclude_dirs=DEFAULT_EXCLUDE_DIRS,
        exclude_paths=(),
    )


def test_default_index_lives_in_the_gitignored_chroma_directory() -> None:
    assert DEFAULT_INDEX_DIR == REPO_ROOT / ".chroma"
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".chroma/" in ignored


def test_missing_index_counts_as_zero_and_is_not_created(tmp_path: Path) -> None:
    index_dir = tmp_path / "chroma"

    assert existing_chunk_count(index_dir) == 0
    assert not index_dir.exists()


def test_empty_collection_counts_as_no_index(tmp_path: Path) -> None:
    """A failed first run can leave the directory behind with nothing in it."""
    index_dir = tmp_path / "chroma"
    ChunkStore(index_dir)

    assert index_dir.exists()
    assert existing_chunk_count(index_dir) == 0


def test_counts_the_chunks_in_an_existing_index(tmp_path: Path, doc_factory) -> None:
    index_dir = tmp_path / "chroma"
    chunks = chunk_document(doc_factory(), REALISTIC)
    ChunkStore(index_dir).upsert(chunks, FakeEmbedder().embed_documents([c.text for c in chunks]))

    assert existing_chunk_count(index_dir) == len(chunks)


def test_gemini_embedder_needs_an_api_key(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
        gemini_embedder(config_for(tmp_path, api_key=None))


def test_gemini_embedder_is_built_with_the_configured_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys: list[str | None] = []

    def fake_client(api_key: str | None) -> object:
        keys.append(api_key)
        return object()

    monkeypatch.setattr(GeminiEmbedder, "_build_client", staticmethod(fake_client))

    embedder = gemini_embedder(config_for(tmp_path, api_key="test-key"))

    assert isinstance(embedder, GeminiEmbedder)
    assert keys == ["test-key"]


def test_cli_uses_the_shared_helpers_rather_than_its_own_copies() -> None:
    assert cli.DEFAULT_INDEX_DIR is DEFAULT_INDEX_DIR
    assert not hasattr(cli, "_gemini_embedder")
    assert not hasattr(cli, "_existing_chunk_count")


# --- excluded projects: fail closed ---------------------------------------------


@pytest.fixture
def indexed_store(tmp_path: Path, knowledge: Path) -> ChunkStore:
    """Macro Tracker, RDBMS and Watch Tracker."""
    store = ChunkStore(tmp_path / "chroma")
    store_chunks(
        collect_chunks(config_for(knowledge, api_key=None)), KeywordEmbedder(), store, rebuild=True
    )
    return store


def test_unset_exclude_projects_is_refused(tmp_path: Path, indexed_store: ChunkStore) -> None:
    """A missing config.local.toml must not quietly expose the private projects."""
    config = config_for(tmp_path, api_key=None)

    with pytest.raises(ConfigError, match=r"\[mcp\] exclude_projects.*config\.local\.toml"):
        excluded_projects(config, indexed_store)


def test_an_explicit_empty_list_excludes_nothing(tmp_path: Path, indexed_store: ChunkStore) -> None:
    config = replace(config_for(tmp_path, api_key=None), mcp_exclude_projects=())

    assert excluded_projects(config, indexed_store) == ()


def test_excluded_names_resolve_to_the_indexed_spelling(
    tmp_path: Path, indexed_store: ChunkStore
) -> None:
    config = replace(
        config_for(tmp_path, api_key=None), mcp_exclude_projects=("rdbms", "WATCH TRACKER")
    )

    assert excluded_projects(config, indexed_store) == ("RDBMS", "Watch Tracker")


def test_a_misspelled_excluded_project_is_refused_without_naming_anything(
    tmp_path: Path, indexed_store: ChunkStore
) -> None:
    """A typo would leave the real project exposed. The message reaches the MCP client,
    so it names neither the typo (close to a private name) nor any indexed project."""
    config = replace(
        config_for(tmp_path, api_key=None), mcp_exclude_projects=("RDBMS", "Macro Trackr")
    )

    with pytest.raises(ConfigError) as exc_info:
        excluded_projects(config, indexed_store)

    message = str(exc_info.value)
    assert "exclude_projects" in message and "config.local.toml" in message
    assert "entry 2 of 2" in message
    for name in ("Macro Trackr", "Macro Tracker", "RDBMS", "Watch Tracker"):
        assert name not in message
