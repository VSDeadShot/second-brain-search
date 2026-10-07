"""One smoke test of the real thing: the installed sbs-mcp script, launched as a
subprocess and spoken to over stdio by the SDK's client - the way Claude Code and
Claude Desktop run it. It proves the console script, Windows stdio framing, and a
stdout that carries nothing but protocol messages.

Gemini is faked at the SDK boundary by tests/stdio_shim/sitecustomize.py, which
Python imports at startup because this test puts its directory on PYTHONPATH. The
shim also points the server at this test's temp config and index, so neither the
owner's .env, config.local.toml nor .chroma is read.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters

from conftest import index_corpus, stub_gemini_embedder

TESTS = Path(__file__).resolve().parent
# A server that writes anything but protocol to stdout leaves the client waiting for
# a reply that never parses. Fail in bounded time instead of hanging the suite.
SESSION_TIMEOUT_SECONDS = 60
SBS_MCP = Path(sys.executable).with_name("sbs-mcp.exe" if os.name == "nt" else "sbs-mcp")


def test_the_installed_sbs_mcp_serves_search_over_stdio(tmp_path: Path, search_corpus: Path) -> None:
    assert SBS_MCP.is_file(), "the sbs-mcp script is missing - run `pip install -e .`"
    index_corpus(search_corpus, tmp_path / "chroma", stub_gemini_embedder())
    (tmp_path / "config.toml").write_text('[mcp]\nexclude_projects = ["Hidden"]\n', encoding="utf-8")
    params = StdioServerParameters(
        command=str(SBS_MCP),
        env={
            "PYTHONPATH": os.pathsep.join([str(TESTS / "stdio_shim"), str(TESTS)]),
            "SBS_TEST_ROOT": str(tmp_path),
            "SBS_SCAN_ROOT": str(search_corpus),
            "GEMINI_API_KEY": "fake-key",
        },
    )

    # The client skips a stdout line that isn't JSON-RPC and hands the parse error to
    # the message handler - so a stray print would otherwise go unnoticed. While
    # serving, mcp 2.x moves the protocol to a private copy of fd 1 and diverts fd 1
    # itself, so the window this guards is startup, before server.run() takes it.
    stray: list[Exception] = []

    async def on_message(message: object) -> None:
        if isinstance(message, Exception):
            stray.append(message)

    async def run():
        with anyio.fail_after(SESSION_TIMEOUT_SECONDS):
            async with Client(params, message_handler=on_message) as client:
                tools = (await client.list_tools()).tools
                listed = await client.call_tool("list_projects", {})
                searched = await client.call_tool("search", {"query": "caching", "k": 20})
                return tools, listed, searched

    tools, listed, searched = anyio.run(run)

    assert stray == [], "sbs-mcp wrote something other than protocol messages to stdout"
    assert {t.name for t in tools} == {"list_projects", "search"}
    assert listed.structured_content == {"projects": ["Alpha", "Beta"]}
    assert not searched.is_error, searched.content[0].text
    results = searched.structured_content["results"]
    assert results
    assert {r["project"] for r in results} <= {"Alpha", "Beta"}
    assert all(r["citation"] == f'{r["project"]} / {r["location"]}' for r in results)
