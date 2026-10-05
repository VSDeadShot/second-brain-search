"""The MCP server, driven in-process through the SDK's own client.

The index is a real Chroma store in tmp_path. Queries are embedded by
KeywordEmbedder, or - for the Gemini error paths - by the server's real
GeminiEmbedder over a stub client, so the retry and wait behaviour under test is
the production code's. No test touches the network. One test reads the owner's
config.local.toml - its [mcp] exclude_projects only, to check the docstring names none
of them - and skips when the file is absent.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client

from second_brain import mcp_server, runtime
from second_brain.config import (
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_INCLUDE_PATTERNS,
    Config,
    ConfigError,
)
from second_brain.embedding import DEFAULT_DIMENSIONS
from second_brain.mcp_server import create_server, server_embedder

from conftest import index_corpus, make_git_chunk, stub_gemini_embedder
from fakes import (
    KeywordEmbedder,
    StubClient,
    StubModels,
    bare_rate_limit_error,
    daily_quota_error,
    rate_limit_error,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PASSAGE_FIELDS = {
    "rank",
    "score",
    "citation",
    "project",
    "location",
    "heading",
    "source",
    "commit",
    "date",
    "date_source",
    "text",
}


def config_for(
    corpus: Path,
    *,
    exclude: tuple[str, ...] | None = ("Hidden",),
    max_searches: int = 50,
    api_key: str | None = "fake-key",
) -> Config:
    return Config(
        scan_root=corpus,
        gemini_api_key=api_key,
        include_patterns=DEFAULT_INCLUDE_PATTERNS,
        include_dirs=("docs",),
        exclude_dirs=DEFAULT_EXCLUDE_DIRS,
        exclude_paths=(),
        mcp_exclude_projects=exclude,
        mcp_max_searches=max_searches,
    )


@pytest.fixture
def keyword_index(tmp_path: Path, search_corpus: Path) -> tuple[Path, KeywordEmbedder]:
    index_dir = tmp_path / "chroma"
    embedder = KeywordEmbedder()
    index_corpus(search_corpus, index_dir, embedder)
    embedder.query_calls.clear()
    return index_dir, embedder


def make_server(corpus: Path, index_dir: Path, embedder: Any, **config: Any):
    return create_server(
        config_for(corpus, **config),
        index_dir=index_dir,
        embedder_factory=lambda _config: embedder,
    )


def calls(server: Any, requests: Sequence[tuple[str, dict[str, Any]]]) -> list[Any]:
    """Every request in ONE client session, in order."""

    async def run() -> list[Any]:
        async with Client(server) as client:
            return [await client.call_tool(name, arguments) for name, arguments in requests]

    return anyio.run(run)


def call(server: Any, tool: str, arguments: dict[str, Any] | None = None) -> Any:
    return calls(server, [(tool, arguments or {})])[0]


def list_tools(server: Any) -> list[Any]:
    async def run() -> list[Any]:
        async with Client(server) as client:
            return (await client.list_tools()).tools

    return anyio.run(run)


def error_text(result: Any) -> str:
    assert result.is_error, result.structured_content
    return result.content[0].text


# --- startup: fail closed ----------------------------------------------------------


def test_server_refuses_to_start_when_exclude_projects_is_unset(
    search_corpus: Path, keyword_index
) -> None:
    index_dir, embedder = keyword_index

    with pytest.raises(ConfigError, match=r"\[mcp\] exclude_projects is not set"):
        make_server(search_corpus, index_dir, embedder, exclude=None)


def test_server_refuses_to_start_when_unset_even_with_no_index(
    tmp_path: Path, search_corpus: Path
) -> None:
    index_dir = tmp_path / "no-index"

    with pytest.raises(ConfigError, match=r"\[mcp\] exclude_projects is not set"):
        make_server(search_corpus, index_dir, KeywordEmbedder(), exclude=None)
    assert not index_dir.exists()


def test_server_refuses_to_start_on_a_misspelled_excluded_project(
    search_corpus: Path, keyword_index
) -> None:
    index_dir, embedder = keyword_index

    with pytest.raises(ConfigError, match="entry 1 of 1"):
        make_server(search_corpus, index_dir, embedder, exclude=("Hiden",))


def test_sbs_mcp_exits_2_with_a_message_and_writes_nothing_to_stdout_when_unset(
    tmp_path: Path,
    search_corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """stdout is the protocol channel: a refusal must go to stderr only."""
    monkeypatch.setattr(runtime, "DEFAULT_INDEX_DIR", tmp_path / "no-index")
    monkeypatch.setattr(mcp_server, "load_config", lambda: config_for(search_corpus, exclude=None))

    with pytest.raises(SystemExit) as exc_info:
        mcp_server.main()

    out, err = capsys.readouterr()
    assert exc_info.value.code == 2
    assert out == ""
    assert "exclude_projects" in err and "config.local.toml" in err


# --- what is exposed ------------------------------------------------------------------


def test_only_list_projects_and_search_are_exposed_both_read_only(
    search_corpus: Path, keyword_index
) -> None:
    index_dir, embedder = keyword_index
    tools = {t.name: t for t in list_tools(make_server(search_corpus, index_dir, embedder))}

    assert set(tools) == {"list_projects", "search"}
    assert all(t.annotations.read_only_hint for t in tools.values())
    k = tools["search"].input_schema["properties"]["k"]
    assert (k["default"], k["minimum"], k["maximum"]) == (5, 1, 20)


def test_list_projects_never_lists_excluded_projects(search_corpus: Path, keyword_index) -> None:
    index_dir, embedder = keyword_index

    result = call(make_server(search_corpus, index_dir, embedder), "list_projects")

    assert result.structured_content == {"result": ["Alpha", "Beta"]}
    assert "Hidden" not in result.content[0].text
    assert embedder.query_calls == []  # free: nothing is embedded


def test_list_projects_matches_exclusions_in_any_case(search_corpus: Path, keyword_index) -> None:
    index_dir, embedder = keyword_index

    result = call(make_server(search_corpus, index_dir, embedder, exclude=("hidden",)), "list_projects")

    assert result.structured_content == {"result": ["Alpha", "Beta"]}


# --- search -----------------------------------------------------------------------


def test_search_returns_full_passages_with_project_location_citations(
    search_corpus: Path, keyword_index
) -> None:
    index_dir, embedder = keyword_index

    result = call(
        make_server(search_corpus, index_dir, embedder),
        "search",
        {"query": "caching layer upstream api dashboard"},
    )

    assert not result.is_error
    top = result.structured_content["results"][0]
    assert set(top) == PASSAGE_FIELDS  # no path, author or author_email
    assert top["rank"] == 1
    assert top["citation"] == "Alpha / README.md"
    assert (top["project"], top["location"], top["source"]) == ("Alpha", "README.md", "doc")
    # The whole passage, not the CLI's 240-character snippet.
    assert len(top["text"]) > 240
    assert top["text"].rstrip().endswith("caching dashboard before users notice them.")


def test_search_returns_5_passages_by_default(search_corpus: Path, keyword_index) -> None:
    index_dir, embedder = keyword_index

    result = call(make_server(search_corpus, index_dir, embedder), "search", {"query": "caching"})

    assert len(result.structured_content["results"]) == 5


def test_k_above_20_is_rejected_before_anything_is_embedded(
    search_corpus: Path, keyword_index
) -> None:
    index_dir, embedder = keyword_index

    result = call(
        make_server(search_corpus, index_dir, embedder), "search", {"query": "caching", "k": 21}
    )

    assert "less than or equal to 20" in error_text(result)
    assert embedder.query_calls == []


def test_search_never_returns_excluded_projects(search_corpus: Path, keyword_index) -> None:
    """Hidden is the best match for this query, so a leak would rank first."""
    index_dir, embedder = keyword_index

    result = call(
        make_server(search_corpus, index_dir, embedder),
        "search",
        {"query": "caching cache warmer", "k": 20},
    )

    projects = {r["project"] for r in result.structured_content["results"]}
    assert projects == {"Alpha", "Beta"}


def test_asking_for_an_excluded_project_is_refused_as_unknown(
    search_corpus: Path, keyword_index
) -> None:
    index_dir, embedder = keyword_index

    result = call(
        make_server(search_corpus, index_dir, embedder),
        "search",
        {"query": "caching", "project": "Hidden"},
    )

    text = error_text(result)
    assert "NOT_ANSWERABLE_AS_ASKED:" in text
    assert "Projects: Alpha, Beta." in text
    assert embedder.query_calls == []


def test_a_commit_is_cited_by_its_hash_and_dated_by_its_author_date(
    search_corpus: Path, keyword_index
) -> None:
    index_dir, embedder = keyword_index
    from second_brain.store import ChunkStore

    chunk = make_git_chunk(project="Alpha")
    ChunkStore(index_dir).upsert([chunk], embedder.embed_documents([chunk.embed_text]))

    result = call(
        make_server(search_corpus, index_dir, embedder),
        "search",
        {"query": "submitReview problem ownership", "k": 1},
    )

    top = result.structured_content["results"][0]
    assert top["citation"] == "Alpha / commit b237453"
    assert (top["source"], top["commit"]) == ("git", "b237453" + "0" * 33)
    assert (top["date"], top["date_source"]) == ("2026-07-02", "committed")


def test_with_no_index_both_tools_say_so_and_create_nothing(
    tmp_path: Path, search_corpus: Path
) -> None:
    index_dir = tmp_path / "no-index"
    server = make_server(search_corpus, index_dir, KeywordEmbedder())

    for result in calls(server, [("list_projects", {}), ("search", {"query": "caching"})]):
        text = error_text(result)
        assert "NOT_ANSWERABLE_AS_ASKED:" in text and "sbs index" in text
    assert not index_dir.exists()


# --- the per-session cap --------------------------------------------------------------


def test_the_51st_search_in_a_session_is_refused_with_session_cap(
    search_corpus: Path, keyword_index
) -> None:
    """Only searches that reach Gemini count: the empty query first is refused before
    embedding and leaves all 50 for real searches."""
    index_dir, embedder = keyword_index
    server = make_server(search_corpus, index_dir, embedder)

    results = calls(server, [("search", {"query": ""})] + [("search", {"query": "caching"})] * 51)

    assert "NOT_ANSWERABLE_AS_ASKED:" in error_text(results[0])
    assert not any(r.is_error for r in results[1:51])
    capped = error_text(results[51])
    assert "SESSION_CAP:" in capped and "50" in capped
    assert len(embedder.query_calls) == 50


def test_the_cap_comes_from_config_and_list_projects_is_free(
    search_corpus: Path, keyword_index
) -> None:
    index_dir, embedder = keyword_index
    server = make_server(search_corpus, index_dir, embedder, max_searches=1)

    results = calls(
        server,
        [("list_projects", {})] * 3 + [("search", {"query": "caching"})] * 2,
    )

    assert not any(r.is_error for r in results[:4])
    assert "SESSION_CAP:" in error_text(results[4])


# --- Gemini errors: one try, no waits, tagged ----------------------------------------


@pytest.fixture
def gemini_index(tmp_path: Path, search_corpus: Path) -> Path:
    """An index in Gemini's vector space, so the server's real embedder may query it."""
    index_dir = tmp_path / "gemini-chroma"
    index_corpus(search_corpus, index_dir, stub_gemini_embedder())
    return index_dir


def gemini_server(corpus: Path, index_dir: Path, models: StubModels, sleeps: list[float], **config: Any):
    return create_server(
        config_for(corpus, **config),
        index_dir=index_dir,
        embedder_factory=lambda cfg: server_embedder(
            cfg, client=StubClient(models), sleep=sleeps.append
        ),
    )


@pytest.mark.parametrize(
    ("error", "tag", "detail"),
    [
        (rate_limit_error("45s"), "RATE_LIMITED: ", "45s"),
        (daily_quota_error(), "QUOTA_DAILY: ", "do not retry"),
        (bare_rate_limit_error(), "RATE_LIMITED_UNEXPLAINED: ", "names no quota"),
    ],
    ids=["per-minute", "daily", "bare-429"],
)
def test_gemini_errors_come_back_tagged_after_one_try_with_no_wait(
    search_corpus: Path, gemini_index: Path, error: Exception, tag: str, detail: str
) -> None:
    models = StubModels(dimensions=DEFAULT_DIMENSIONS, raise_queue=[error] * 3)
    sleeps: list[float] = []

    result = call(gemini_server(search_corpus, gemini_index, models, sleeps), "search", {"query": "caching"})

    text = error_text(result)
    assert tag in text
    assert detail in text.lower()
    assert len(models.calls) == 1
    assert sleeps == []


def test_a_working_gemini_search_costs_one_call(search_corpus: Path, gemini_index: Path) -> None:
    models = StubModels(dimensions=DEFAULT_DIMENSIONS)
    sleeps: list[float] = []

    result = call(gemini_server(search_corpus, gemini_index, models, sleeps), "search", {"query": "caching"})

    assert not result.is_error
    assert len(models.calls) == 1 and sleeps == []


def test_a_missing_api_key_is_a_config_error_and_list_projects_still_works(
    search_corpus: Path, gemini_index: Path
) -> None:
    server = create_server(
        config_for(search_corpus, api_key=None), index_dir=gemini_index, embedder_factory=server_embedder
    )

    listed, searched = calls(server, [("list_projects", {}), ("search", {"query": "caching"})])

    assert listed.structured_content == {"result": ["Alpha", "Beta"]}
    text = error_text(searched)
    assert "CONFIG:" in text and "GEMINI_API_KEY" in text


# --- packaging and docs ------------------------------------------------------------


def test_mcp_is_pinned_exactly_and_sbs_mcp_is_installed_as_a_script() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert "mcp==2.3.0" in project["dependencies"]
    assert project["scripts"]["sbs-mcp"] == "second_brain.mcp_server:main"


# Generic shapes, so this test names nothing private itself.
ABSOLUTE_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def test_registration_steps_use_placeholders_only() -> None:
    doc = mcp_server.__doc__ or ""

    assert "claude mcp add second-brain --scope user" in doc
    assert "claude_desktop_config.json" in doc
    assert "<repo>" in doc
    assert ABSOLUTE_WINDOWS_PATH.findall(doc) == []
    assert EMAIL.findall(doc) == []


def test_registration_steps_name_no_excluded_project() -> None:
    """The private names live only in the gitignored config.local.toml, so they are
    read from there - the one test that reads it, and only its [mcp] table."""
    local = REPO_ROOT / "config.local.toml"
    if not local.is_file():
        pytest.skip("config.local.toml is absent (e.g. a fresh clone): no private names to check")
    mcp_table = tomllib.loads(local.read_text(encoding="utf-8")).get("mcp", {})
    excluded = mcp_table.get("exclude_projects")
    if not excluded:
        pytest.skip("config.local.toml sets no [mcp] exclude_projects: no private names to check")

    doc = (mcp_server.__doc__ or "").lower()

    # Reported by position, so a failure doesn't print the private name either.
    named = [i for i, name in enumerate(excluded, start=1) if name.lower() in doc]
    assert named == [], f"exclude_projects entries {named} appear in the docstring"
