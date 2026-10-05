"""Query plumbing shared by every front end - the `sbs` CLI and the MCP server.

Where the index lives, whether one exists yet, how the real Gemini embedder is
built, and which projects the MCP server must hide. Kept apart from cli.py so a second front end reuses these instead of
copying them, and imports nothing from click.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .config import Config, ConfigError
from .embedding import Embedder, GeminiEmbedder
from .store import ChunkStore

# src/second_brain/runtime.py -> repo root. Resolved from this file, not the
# working directory, so a client that starts us elsewhere still finds the index.
REPO_ROOT = Path(__file__).resolve().parents[2]
# Gitignored.
DEFAULT_INDEX_DIR = REPO_ROOT / ".chroma"

EmbedderFactory = Callable[[Config], Embedder]


def gemini_embedder(config: Config) -> Embedder:
    return GeminiEmbedder(api_key=config.gemini_api_key)


def existing_chunk_count(index_dir: Path) -> int:
    """Chunks already in the index - 0 when there is none. Never creates the directory,
    and treats an empty collection (e.g. left by a failed first run) as no index."""
    if not index_dir.exists():
        return 0
    return ChunkStore(index_dir).count()


def excluded_projects(config: Config, store: ChunkStore) -> tuple[str, ...]:
    """The projects the MCP server hides, spelled as the index spells them.

    Fails closed. Unset means config.local.toml is missing or incomplete, and an
    entry matching no indexed project is most likely a typo - either way the project
    it meant to hide would be exposed. The message can reach the MCP client, so it
    names neither the entry nor any project: a typo sits one letter from a private name.
    """
    names = config.mcp_exclude_projects
    if names is None:
        raise ConfigError(
            "[mcp] exclude_projects is not set. Set it in config.local.toml to the "
            "projects to hide, or to [] to hide none."
        )
    indexed = {name.lower(): name for name in store.projects()}
    resolved: list[str] = []
    for position, name in enumerate(names, start=1):
        match = indexed.get(name.strip().lower())
        if match is None:
            raise ConfigError(
                f"[mcp] exclude_projects entry {position} of {len(names)} "
                "(config.local.toml) matches no indexed project, so the project it "
                "means to hide would be exposed. Check its spelling against "
                "`sbs discover`."
            )
        resolved.append(match)
    return tuple(resolved)
