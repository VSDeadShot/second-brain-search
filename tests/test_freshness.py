"""Dating a document: what git says, and what to do when git can't say.

The fake runner proves the fallbacks; only the real-git test proves the command
itself is right, so both are here on purpose.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from second_brain.freshness import (
    GIT_TIMEOUT_SECONDS,
    DateLookup,
    FileDate,
)

from conftest import make_file
from fakes import FakeRunner


@pytest.fixture
def doc(tmp_path: Path) -> Path:
    return make_file(tmp_path / "proj" / "README.md", "# Doc\n")


def mtime_date(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


# --- the happy path -------------------------------------------------------------


def test_the_date_comes_from_gits_last_commit(doc: Path) -> None:
    lookup = DateLookup(runner=FakeRunner("2026-08-18\n"))

    assert lookup.date_for(doc) == FileDate("2026-08-18", "git")


def test_git_is_run_as_an_argument_list_never_through_a_shell(doc: Path) -> None:
    runner = FakeRunner()
    DateLookup(runner=runner).date_for(doc)
    args, kwargs = runner.calls[0]

    assert args == [
        "git",
        "-C",
        str(doc.parent),
        "log",
        "-1",
        "--format=%cs",
        "--",
        str(doc),
    ]
    assert isinstance(args, list)
    assert not kwargs.get("shell", False)
    assert kwargs["timeout"] == GIT_TIMEOUT_SECONDS


# --- every way git can fail falls back to the file's own mtime ------------------


def test_git_missing_falls_back_to_mtime(doc: Path) -> None:
    lookup = DateLookup(runner=FakeRunner(raises=FileNotFoundError("git")))

    assert lookup.date_for(doc) == FileDate(mtime_date(doc), "mtime")


def test_git_timing_out_falls_back_to_mtime(doc: Path) -> None:
    timeout = subprocess.TimeoutExpired(cmd=["git"], timeout=GIT_TIMEOUT_SECONDS)
    lookup = DateLookup(runner=FakeRunner(raises=timeout))

    assert lookup.date_for(doc) == FileDate(mtime_date(doc), "mtime")


def test_git_erroring_falls_back_to_mtime(doc: Path) -> None:
    """Not a repository, or any other non-zero exit."""
    lookup = DateLookup(runner=FakeRunner(stdout="", returncode=128))

    assert lookup.date_for(doc) == FileDate(mtime_date(doc), "mtime")


def test_an_untracked_file_falls_back_to_mtime(doc: Path) -> None:
    """git exits 0 with no output when the file is in a repo but never committed."""
    lookup = DateLookup(runner=FakeRunner(stdout="\n", returncode=0))

    assert lookup.date_for(doc) == FileDate(mtime_date(doc), "mtime")


def test_an_unreadable_date_falls_back_to_mtime(doc: Path) -> None:
    lookup = DateLookup(runner=FakeRunner(stdout="last tuesday\n"))

    assert lookup.date_for(doc) == FileDate(mtime_date(doc), "mtime")


def test_a_file_that_is_gone_is_dated_unknown(tmp_path: Path) -> None:
    """Indexed, then deleted: nothing can date it, and nothing may crash."""
    lookup = DateLookup(runner=FakeRunner(stdout="", returncode=128))

    assert lookup.date_for(tmp_path / "vanished.md") == FileDate("unknown", "unknown")


# --- one git call per file ------------------------------------------------------


def test_the_same_file_is_only_asked_about_once(doc: Path) -> None:
    runner = FakeRunner()
    lookup = DateLookup(runner=runner)

    lookup.date_for(doc)
    lookup.date_for(doc)

    assert len(runner.calls) == 1


def test_different_files_are_each_asked_about(tmp_path: Path) -> None:
    runner = FakeRunner()
    lookup = DateLookup(runner=runner)

    lookup.date_for(make_file(tmp_path / "a.md"))
    lookup.date_for(make_file(tmp_path / "b.md"))

    assert len(runner.calls) == 2


# --- changed since indexed ------------------------------------------------------


def test_a_file_edited_after_indexing_is_flagged(doc: Path) -> None:
    lookup = DateLookup(runner=FakeRunner())

    assert lookup.changed_since_indexed(doc, doc.stat().st_mtime - 3600)


def test_a_file_untouched_since_indexing_is_not_flagged(doc: Path) -> None:
    lookup = DateLookup(runner=FakeRunner())

    assert not lookup.changed_since_indexed(doc, doc.stat().st_mtime)


def test_a_sub_second_difference_is_not_a_change(doc: Path) -> None:
    """Filesystems and Chroma round mtimes differently; a whisker is not an edit."""
    lookup = DateLookup(runner=FakeRunner())

    assert not lookup.changed_since_indexed(doc, doc.stat().st_mtime - 0.4)


def test_a_missing_file_is_not_reported_as_changed(tmp_path: Path) -> None:
    lookup = DateLookup(runner=FakeRunner())

    assert not lookup.changed_since_indexed(tmp_path / "vanished.md", 1.0)


# --- against real git -----------------------------------------------------------


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_a_real_commit_date_is_read_from_a_real_repo(tmp_path: Path) -> None:
    """The fake runner cannot prove the argv is right. This can."""
    repo = tmp_path / "repo"
    doc = make_file(repo / "README.md", "# Real\n")
    run = lambda *args: subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    run("init")
    run("config", "user.email", "test@example.invalid")
    run("config", "user.name", "Test")
    run("add", "README.md")
    run("commit", "-m", "add readme")

    result = DateLookup().date_for(doc)

    assert result == FileDate(datetime.now().strftime("%Y-%m-%d"), "git")
