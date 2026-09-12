"""Full-rebuild indexing: discover -> read -> chunk -> embed -> store.

SPEC decision 5 makes every index a full rebuild, so this deliberately has no
incremental logic. Deterministic chunk ids make that cheap: upserts overwrite
rather than accumulate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, Chunk, chunk_document
from .config import Config
from .discovery import DiscoveredDoc, discover_documents
from .embedding import Embedder
from .store import ChunkStore


@dataclass(frozen=True)
class IndexReport:
    documents: int
    chunks: int
    projects: int
    skipped: list[tuple[str, str]] = field(default_factory=list)


def index_documents(
    config: Config,
    embedder: Embedder,
    store: ChunkStore,
    *,
    docs: Sequence[DiscoveredDoc] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> IndexReport:
    """Embed and store every discovered document.

    `docs` can be supplied to reuse an existing discovery pass; otherwise the
    scan runs here.
    """
    documents = list(docs) if docs is not None else discover_documents(config)

    chunks: list[Chunk] = []
    skipped: list[tuple[str, str]] = []
    projects: set[str] = set()
    read_count = 0

    for doc in documents:
        try:
            # Self-authored markdown should be clean UTF-8; replace rather than
            # abort so one bad byte cannot stop a whole rebuild.
            text = doc.path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            skipped.append((str(doc.path), str(exc)))
            continue

        read_count += 1
        projects.add(doc.project)
        chunks.extend(chunk_document(doc, text, max_chars=max_chars, overlap=overlap))

    if chunks:
        vectors = embedder.embed_documents([c.text for c in chunks])
        store.upsert(chunks, vectors)

    return IndexReport(
        documents=read_count,
        chunks=len(chunks),
        projects=len(projects),
        skipped=skipped,
    )
