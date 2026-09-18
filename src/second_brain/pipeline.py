"""Full-rebuild indexing: discover -> read -> chunk -> embed -> store.

SPEC decision 5 makes every index a full rebuild. The index is rebuilt from
scratch each time, but embeddings are not re-bought: vectors are saved by
content hash as each batch returns, so a run that fails partway keeps its work
and the next run embeds only text it has not seen. That matters because the
Gemini free tier allows 1000 embedded texts a day and a rebuild needs ~812.

Chunking is split out as `collect_chunks` so a dry run can report exactly what
indexing would embed without calling the API.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from .chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, Chunk, chunk_document
from .config import Config
from .discovery import DiscoveredDoc, discover_documents
from .embedding import BatchCallback, Embedder, EmbeddingError
from .embedding_cache import EmbeddingCache, cache_key
from .store import ChunkStore, VectorSpaceMismatch


@dataclass(frozen=True)
class IndexReport:
    documents: int
    chunks: int
    projects: int
    skipped: list[tuple[str, str]] = field(default_factory=list)
    chunks_per_project: dict[str, int] = field(default_factory=dict)
    # Unique texts sent to the embedding API on this run.
    embedded: int = 0
    # Chunks whose vector came from a saved embedding instead.
    reused: int = 0


@dataclass(frozen=True)
class EmbeddingPlan:
    reused: int
    """Chunks whose embedding is already saved."""
    to_embed: int
    """Unique texts that would be sent to the API - identical text is embedded once."""
    pending: tuple[str, ...] = field(default=(), compare=False, repr=False)
    """Those texts, in send order - batch count depends on their size, not just count."""


@dataclass(frozen=True)
class ChunkCollection:
    chunks: list[Chunk]
    report: IndexReport


def _keys(chunks: Sequence[Chunk], namespace: str) -> list[str]:
    return [cache_key(namespace, chunk.content_hash) for chunk in chunks]


def embedding_plan(
    chunks: Sequence[Chunk], namespace: str, cache: EmbeddingCache | None
) -> EmbeddingPlan:
    """How much of a run is already paid for. Read-only."""
    keys = _keys(chunks, namespace)
    saved = cache.get_many(keys) if cache is not None and keys else {}
    pending: dict[str, str] = {}
    for key, chunk in zip(keys, chunks):
        if key not in saved:
            pending.setdefault(key, chunk.embed_text)
    return EmbeddingPlan(
        reused=sum(1 for key in keys if key in saved),
        to_embed=len(pending),
        pending=tuple(pending.values()),
    )


def collect_chunks(
    config: Config,
    *,
    docs: Sequence[DiscoveredDoc] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> ChunkCollection:
    """Discover, read and chunk - everything short of embedding.

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

    report = IndexReport(
        documents=read_count,
        chunks=len(chunks),
        projects=len(projects),
        skipped=skipped,
        chunks_per_project=dict(Counter(c.project for c in chunks)),
    )
    return ChunkCollection(chunks=chunks, report=report)


def store_chunks(
    collected: ChunkCollection,
    embedder: Embedder,
    store: ChunkStore,
    *,
    rebuild: bool = False,
    on_batch: BatchCallback | None = None,
    cache: EmbeddingCache | None = None,
) -> IndexReport:
    """Embed an already-collected set of chunks and write them to the store.

    Upsert alone never removes anything: chunk ids are deterministic, so a
    deleted document - or the tail of a shrunk one - would stay searchable.
    `rebuild=True` makes the store match this run exactly. Every vector is in
    hand BEFORE the reset, so an API failure leaves the previous index intact.

    With a `cache`, each batch's vectors are saved the moment they return -
    independently of the index - and texts already saved are not re-embedded.
    """
    chunks = collected.chunks
    namespace = embedder.cache_namespace

    # Checked before embedding, so refusing costs no quota. A rebuild replaces
    # the whole index, so switching embedders is legitimate there.
    if not rebuild and store.namespace not in (None, namespace):
        raise VectorSpaceMismatch(
            f"This index was built with {store.namespace!r} but the embedder is "
            f"{namespace!r}; mixing them would make search results meaningless. "
            "Rebuild the index instead."
        )

    keys = _keys(chunks, namespace)
    known = cache.get_many(keys) if cache is not None and keys else {}
    reused = sum(1 for key in keys if key in known)

    # Unique missing texts, in first-seen order.
    pending: dict[str, str] = {}
    for key, chunk in zip(keys, chunks):
        if key not in known:
            pending.setdefault(key, chunk.embed_text)
    pending_keys = list(pending)

    def save_batch(offset: int, batch_vectors: list[list[float]]) -> None:
        batch = list(zip(pending_keys[offset : offset + len(batch_vectors)], batch_vectors))
        if cache is not None:
            cache.put_many(batch)
        known.update(batch)

    if pending_keys:
        returned = embedder.embed_documents(
            list(pending.values()), on_batch=on_batch, on_embedded=save_batch
        )
        if len(returned) != len(pending_keys):
            raise EmbeddingError(
                f"Embedder returned {len(returned)} vectors for {len(pending_keys)} texts."
            )
        known.update(zip(pending_keys, returned))

    vectors = [known[key] for key in keys]

    if rebuild:
        store.reset(namespace=namespace)
    elif store.namespace is None:
        store.record_namespace(namespace)
    store.upsert(chunks, vectors)

    return replace(collected.report, embedded=len(pending_keys), reused=reused)


def index_documents(
    config: Config,
    embedder: Embedder,
    store: ChunkStore,
    *,
    docs: Sequence[DiscoveredDoc] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    rebuild: bool = False,
    cache: EmbeddingCache | None = None,
) -> IndexReport:
    """Collect, embed and store every discovered document in one call."""
    collected = collect_chunks(config, docs=docs, max_chars=max_chars, overlap=overlap)
    return store_chunks(collected, embedder, store, rebuild=rebuild, cache=cache)
