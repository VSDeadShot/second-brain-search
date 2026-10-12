"""Live tests - the ones that call Gemini - run only when asked for with --live.

An environment-variable switch is not enough: conftest.py loads .env, and in PowerShell
`$env:GEMINI_API_KEY = ""` deletes the variable, so .env put the key back and the live
test spent quota anyway. A plain `pytest` must make no Gemini call, in any shell.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

TESTS = Path(__file__).resolve().parent

A_LIVE_TEST = """
import pytest


@pytest.mark.live
def test_calls_gemini():
    raise AssertionError("a live test ran")


def test_offline():
    pass
"""


def our_markers() -> list[str]:
    raw = tomllib.loads((TESTS.parent / "pyproject.toml").read_text(encoding="utf-8"))
    return raw["tool"]["pytest"]["ini_options"]["markers"]


def run_with_our_conftest(pytester: pytest.Pytester, *args: str) -> pytest.RunResult:
    markers = ", ".join(repr(m) for m in our_markers())
    pytester.makepyprojecttoml(f"[tool.pytest.ini_options]\nmarkers = [{markers}]\n")
    pytester.makeconftest((TESTS / "conftest.py").read_text(encoding="utf-8"))
    pytester.makepyfile(A_LIVE_TEST)
    return pytester.runpytest("-rs", "--strict-markers", *args)


def test_a_plain_run_skips_live_tests_and_says_how_to_run_them(pytester: pytest.Pytester) -> None:
    result = run_with_our_conftest(pytester)

    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.fnmatch_lines(["*--live*"])


def test_live_tests_run_when_asked_for(pytester: pytest.Pytester) -> None:
    result = run_with_our_conftest(pytester, "--live")

    result.assert_outcomes(passed=1, failed=1)


def test_the_live_marker_is_registered() -> None:
    assert any(m.startswith("live:") for m in our_markers())


def test_the_gemini_round_trip_is_a_live_test() -> None:
    from test_embedding import test_live_gemini_embedding_roundtrip

    marks = {m.name for m in getattr(test_live_gemini_embedding_roundtrip, "pytestmark", [])}
    assert "live" in marks
