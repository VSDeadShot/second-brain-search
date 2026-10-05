"""Query plumbing shared by every front end - the `sbs` CLI and the MCP server.

Where the index lives, whether one exists yet, and how the real Gemini embedder
is built. Kept apart from cli.py so a second front end reuses these instead of
copying them, and imports nothing from click.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .config import Config
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
