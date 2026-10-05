"""Test-only. Python imports this at startup when its directory is on PYTHONPATH -
which only tests/test_mcp_stdio.py arranges, for the sbs-mcp process it spawns.

With SBS_TEST_ROOT set it fakes Gemini at the SDK boundary (google.genai.Client)
and points the server at that directory's config.toml and chroma/ index. The real
script, entry point, server and stdio transport all run unmodified; nothing in
src/ knows this file exists. It must never write to stdout - stdout is the
protocol channel.
"""

import os
from pathlib import Path

_root = os.environ.get("SBS_TEST_ROOT")

if _root:
    from google import genai

    import fakes
    from second_brain import config, runtime
    from second_brain.embedding import DEFAULT_DIMENSIONS

    def _stub_client(*_args, **_kwargs):
        return fakes.StubClient(fakes.StubModels(dimensions=DEFAULT_DIMENSIONS))

    genai.Client = _stub_client
    config._repo_root = lambda: Path(_root)
    runtime.DEFAULT_INDEX_DIR = Path(_root) / "chroma"
