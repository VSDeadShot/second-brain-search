"""Full-rebuild indexing: discover -> read -> chunk -> embed -> store.

SPEC decision 5 makes every index a full rebuild. The index is rebuilt from
scratch each time, but embeddings are not re-bought: vectors are saved by
content hash as each batch returns, so a run that fails partway keeps its work
and the next run embeds only text it has not seen. That matters because the
Gemini free tier allows 1000 embedded texts a day and a rebuild needs ~812.

Chunking is split out as `collect_chunks` so a dry run can report exactly what
indexing would embed without calling the API. With [git] enabled it also reads
each discovered repo's commit history (git_history.py) - only the configured
authors' commits, and it refuses to run with no author configured.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from .chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, Chunk, chunk_document
from .config import Config, ConfigError
from .discovery import DiscoveredDoc, discover_documents
from .embedding import BatchCallback, Embedder, EmbeddingError
from .embedding_cache import EmbeddingCache, cache_key
from .git_history import GitError, RepoHistory, RepoStatus, collect_repo_history, repo_status
from .store import ChunkStore, VectorSpaceMismatch


@dataclass(frozen=True)
class GitRepoReport:
    """One repo's history: what was read, from where, and what was kept."""

    project: str
    ref: str
    branch: str | None
    flags: tuple[str, ...]
    commits: int
    kept: int
    merges: int
    version_bumps: int
    other_authors: int
    with_excerpt: int
    chunks: int


@dataclass(frozen=True)
class IndexReport:
    documents: int
    chunks: int
    """Every chunk, docs and commits."""
    projects: int
    skipped: list[tuple[str, str]] = field(default_factory=list)
    chunks_per_project: dict[str, int] = field(default_factory=dict)
    # Unique texts sent to the embedding API on this run.
    embedded: int = 0
    # Chunks whose vector came from a saved embedding instead.
    reused: int = 0
    git_repos: list[GitRepoReport] = field(default_factory=list)
    git_chunks: int = 0


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


def require_git_authors(config: Config) -> None:
    """Refuse a git-enabled run that names no author.

    An empty author list keeps nothing, so the run would look fine while quietly
    indexing no history - or, if matching ever loosened, index everyone's.
    """
    if config.git_enabled and not (config.git_author_emails or config.git_author_names):
        raise ConfigError(
            "Git history is enabled ([git] enabled = true) but no author is set: "
            "[git] author_emails and author_names are both empty. Add your commit "
            "email(s) or name(s) to config.local.toml (gitignored) - only those "
            "authors' commits are indexed - or set [git] enabled = false."
        )


def _git_repos(documents: Sequence[DiscoveredDoc]) -> list[tuple[str, Path]]:
    """(project, repo root) for each discovered project that is a git repo."""
    roots = {doc.project_root: doc.project for doc in documents}
    repos = [(project, root) for root, project in roots.items() if (root / ".git").exists()]
    return sorted(repos, key=lambda pair: pair[0].lower())


def _repo_report(project: str, status: RepoStatus, history: RepoHistory) -> GitRepoReport:
    return GitRepoReport(
        project=project,
        ref=status.ref,
        branch=status.branch,
        flags=tuple(status.flags),
        commits=history.commits,
        kept=history.kept,
        merges=history.merges,
        version_bumps=history.version_bumps,
        other_authors=history.other_authors,
        with_excerpt=history.with_excerpt,
        chunks=len(history.chunks),
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
    require_git_authors(config)
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

    git_repos: list[GitRepoReport] = []
    git_chunk_count = 0
    if config.git_enabled:
        for project, repo in _git_repos(documents):
            try:
                status = repo_status(repo)
                history = collect_repo_history(
                    repo,
                    project=project,
                    ref=status.ref,
                    author_emails=config.git_author_emails,
                    author_names=config.git_author_names,
                    thin_message_chars=config.git_thin_message_chars,
                    diff_excerpt_chars=config.git_diff_excerpt_chars,
                    max_chars=max_chars,
                    overlap=overlap,
                )
            except GitError as exc:
                # A repo git can't read loses its history, not its docs.
                skipped.append((str(repo), f"git history: {exc}"))
                continue
            projects.add(project)
            chunks.extend(history.chunks)
            git_chunk_count += len(history.chunks)
            git_repos.append(_repo_report(project, status, history))

    report = IndexReport(
        documents=read_count,
        chunks=len(chunks),
        projects=len(projects),
        skipped=skipped,
        chunks_per_project=dict(Counter(c.project for c in chunks)),
        git_repos=git_repos,
        git_chunks=git_chunk_count,
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
