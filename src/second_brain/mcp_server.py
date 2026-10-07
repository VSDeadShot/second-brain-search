r"""MCP server: lets Claude Code and Claude Desktop search the second brain.

`sbs-mcp` speaks MCP over stdio. The client starts it, exchanges JSON-RPC on its
stdin/stdout, and stops it when the session ends - no port, no listener.

Two read-only tools over the index that `sbs index` builds:
- list_projects - the project names that can be searched. Free.
- search - the k most relevant passages (5 by default, at most 20), each with its
  full text and a "Project / location" citation. One Gemini embedding per call.

Nothing is generated here: Claude writes the answer from the passages. The server
refuses to start until `[mcp] exclude_projects` is set in the gitignored
config.local.toml (`exclude_projects = []` hides nothing), and leaves the projects
listed there out of both tools. A session may run `[mcp] max_searches` searches (50
by default), and a Gemini rate limit fails fast rather than waiting it out.

Registration on Windows - replace <repo> with this repository's folder.

Claude Code, available in every project:

    claude mcp add second-brain --scope user -- "<repo>\.venv\Scripts\sbs-mcp.exe"

Check it with `claude mcp list`, or `/mcp` inside a session. Don't use
`--scope project`: that writes the absolute path into a committed .mcp.json.

Claude Desktop: Settings -> Developer -> Edit Config opens
claude_desktop_config.json. Merge this in, keeping what is already there:

    "mcpServers": {
      "second-brain": {"command": "<repo>\\.venv\\Scripts\\sbs-mcp.exe", "args": []}
    }

then quit Desktop from the system tray and start it again. No "env" block is
needed: .env and the config files are found from the package's own location, not
from the directory the client starts the server in.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from . import runtime
from .config import Config, ConfigError, load_config
from .embedding import (
    DailyQuotaExceeded,
    Embedder,
    EmbeddingError,
    GeminiEmbedder,
    UnexplainedRateLimit,
)
from .freshness import DateLookup, chunk_date
from .gemini_errors import rate_limit_delay
from .retrieval import DEFAULT_K, RetrievalError, RetrievedChunk, retrieve
from .runtime import EmbedderFactory, excluded_projects, existing_chunk_count, require_exclusions
from .store import ChunkStore

MAX_K = 20
NO_INDEX = "There is no index yet - run `sbs index` first."

READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)

INSTRUCTIONS = (
    "Searches the owner's own project documentation and git history. Use `search` to "
    "find passages and answer from them, citing each by its 'Project / location'. "
    "Use `list_projects` for exact project names."
)

LIST_PROJECTS_DESCRIPTION = (
    "The project names that can be searched - pass one as `project` to `search` to "
    "search only that project. Free: nothing is embedded."
)

SEARCH_DESCRIPTION = (
    "Semantic search over the owner's own project documentation and git history "
    "(READMEs, CLAUDE.md, specs, changelogs, and commit messages). Returns the k most "
    "relevant passages, best first, each with its full text and a citation "
    "'Project / location' - a file path, or 'commit <hash>'. Answer from these "
    "passages and cite them by that citation. A passage being returned does not mean "
    "it answers the question: if none does, say so rather than guessing. Each call "
    "costs one Gemini embedding; this session allows {max_searches}."
)


class Passage(BaseModel):
    rank: int
    score: float = Field(description="Cosine similarity to the query; higher is closer.")
    citation: str = Field(description="'Project / location' - cite the passage by this.")
    project: str
    location: str = Field(description="A file path in the project, or 'commit <hash>'.")
    heading: str = Field(description="The heading path within the file, or a commit subject.")
    source: str = Field(description="'doc' or 'git'.")
    commit: str = Field(description="The full hash for a commit; empty for a doc.")
    date: str = Field(description="YYYY-MM-DD, or 'unknown'.")
    date_source: str = Field(description="'committed', 'git', 'mtime' or 'unknown'.")
    text: str = Field(description="The full passage.")


class SearchResults(BaseModel):
    query: str
    results: list[Passage]


class Projects(BaseModel):
    # A model, not a bare list: a list is sent as one text block per name, and Claude
    # Desktop joins blocks with nothing between them.
    projects: list[str]


class SessionCapReached(Exception):
    """This server process has sent its `[mcp] max_searches` queries to Gemini."""


def server_embedder(
    config: Config,
    *,
    client: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Embedder:
    """The real Gemini embedder, trying each query once. The CLI retries a rate limit
    after waiting out the server's delay (~60s a time); here that would hold the
    tool call open for minutes, so the delay goes back to the client instead."""
    return GeminiEmbedder(api_key=config.gemini_api_key, max_retries=1, client=client, sleep=sleep)


class _SearchBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._used = 0
        self._lock = threading.Lock()  # tools run on worker threads

    def spend(self) -> None:
        with self._lock:
            if self._used >= self.limit:
                raise SessionCapReached(
                    f"This session has used all {self.limit} of its searches (each costs "
                    "one Gemini embedding). Start a new session to search again; the limit "
                    "is [mcp] max_searches."
                )
            self._used += 1


class _BudgetedEmbedder:
    """Charges the session budget at the moment a query would reach Gemini, so a
    search refused before embedding (empty query, unknown project) costs nothing."""

    def __init__(self, inner: Embedder, budget: _SearchBudget) -> None:
        self._inner = inner
        self._budget = budget

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def cache_namespace(self) -> str:
        return self._inner.cache_namespace

    def embed_documents(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        return self._inner.embed_documents(texts, **kwargs)

    def embed_query(self, text: str) -> list[float]:
        self._budget.spend()
        return self._inner.embed_query(text)


def _tool_error(exc: Exception) -> ToolError:
    """One tag per kind of failure, so the client can tell whether trying again helps.
    Messages are the existing ones; anything unforeseen is left to the SDK, which
    reports it without a traceback."""
    if isinstance(exc, SessionCapReached):
        return ToolError(f"SESSION_CAP: {exc}")
    if isinstance(exc, ConfigError):
        return ToolError(f"CONFIG: {exc}")
    if isinstance(exc, RetrievalError):
        return ToolError(f"NOT_ANSWERABLE_AS_ASKED: {exc}")
    if isinstance(exc, DailyQuotaExceeded):
        return ToolError(f"QUOTA_DAILY: {exc} Do not retry today.")
    if isinstance(exc, UnexplainedRateLimit):
        return ToolError(f"RATE_LIMITED_UNEXPLAINED: {exc}")
    # A per-minute 429 that the single try gave up on: the 429 is the cause.
    delay = rate_limit_delay(exc.__cause__) if exc.__cause__ is not None else None
    if delay is not None:
        return ToolError(f"RATE_LIMITED: Gemini is rate limiting embeddings. Try again in {delay:.0f}s.")
    return ToolError(f"GEMINI_ERROR: {exc}")


def _passage(rank: int, chunk: RetrievedChunk, dates: DateLookup) -> Passage:
    dated = chunk_date(chunk, dates)
    return Passage(
        rank=rank,
        score=chunk.score,
        citation=f"{chunk.project} / {chunk.rel_path}",
        project=chunk.project,
        location=chunk.rel_path,
        heading=chunk.heading_path,
        source=chunk.source,
        commit=chunk.commit,
        date=dated.date,
        date_source=dated.source,
        text=chunk.text,
    )


def create_server(
    config: Config,
    *,
    index_dir: Path | None = None,
    embedder_factory: EmbedderFactory = server_embedder,
    dates_factory: Callable[[], DateLookup] = DateLookup,
) -> MCPServer:
    """Build the server, or raise ConfigError so it never starts: when `[mcp]
    exclude_projects` is unset, or (with an index) names a project that isn't in it."""
    index_dir = index_dir if index_dir is not None else runtime.DEFAULT_INDEX_DIR
    require_exclusions(config)
    if existing_chunk_count(index_dir):
        excluded_projects(config, ChunkStore(index_dir))

    budget = _SearchBudget(config.mcp_max_searches)
    server = MCPServer(name="second-brain", instructions=INSTRUCTIONS, log_level="WARNING")

    def open_store() -> ChunkStore:
        # Checked before opening, which would otherwise create the index directory.
        if existing_chunk_count(index_dir) == 0:
            raise RetrievalError(NO_INDEX)
        return ChunkStore(index_dir)

    @server.tool(description=LIST_PROJECTS_DESCRIPTION, annotations=READ_ONLY)
    def list_projects() -> Projects:
        try:
            store = open_store()
            hidden = excluded_projects(config, store)
        except (ConfigError, RetrievalError) as exc:
            raise _tool_error(exc) from exc
        return Projects(projects=[name for name in store.projects() if name not in hidden])

    @server.tool(
        description=SEARCH_DESCRIPTION.format(max_searches=config.mcp_max_searches),
        annotations=READ_ONLY,
    )
    def search(
        query: Annotated[str, Field(description="What to look for, in plain words.")],
        k: Annotated[int, Field(ge=1, le=MAX_K, description="How many passages to return.")] = DEFAULT_K,
        project: Annotated[
            str | None, Field(description="Only search this project (any case).")
        ] = None,
    ) -> SearchResults:
        try:
            store = open_store()
            hidden = excluded_projects(config, store)
            try:
                embedder = embedder_factory(config)
            except EmbeddingError as exc:  # no GEMINI_API_KEY
                raise ConfigError(str(exc)) from exc
            chunks = retrieve(
                query,
                _BudgetedEmbedder(embedder, budget),
                store,
                k=k,
                project=project,
                exclude_projects=hidden,
            )
        except (SessionCapReached, ConfigError, RetrievalError, EmbeddingError) as exc:
            raise _tool_error(exc) from exc

        dates = dates_factory()  # per call: a long-lived cache would serve stale dates
        return SearchResults(
            query=query,
            results=[_passage(rank, chunk, dates) for rank, chunk in enumerate(chunks, start=1)],
        )

    return server


def main() -> None:
    """Console-script entry point. A refusal goes to stderr: stdout is the protocol."""
    try:
        server = create_server(load_config())
    except ConfigError as exc:
        print(f"sbs-mcp: not starting - {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    server.run("stdio")
