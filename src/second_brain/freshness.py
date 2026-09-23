"""How old is a document, and has it changed since it was indexed?

A citation is worth less without a date: the corpus holds docs that were true
when written and aren't now. git's last commit date is the honest answer where
it exists, so that is asked for first and the answer is labelled with where it
came from - a reader can tell "git says August" from "the file was touched in
August".

git is consulted through a plain argument list with a timeout, never a shell.
Every way it can fail - not installed, not a repository, file never committed,
hanging - falls back to the file's own mtime, because a date is a nicety and an
answer that crashes is not.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

GIT_TIMEOUT_SECONDS = 5.0
DATE_FORMAT = "%Y-%m-%d"
# Chroma stores mtimes as floats and filesystems round them differently, so a
# sub-second difference is noise, not an edit.
CHANGE_TOLERANCE_SECONDS = 1.0

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class FileDate:
    date: str
    """YYYY-MM-DD, or "unknown" when nothing could date the file."""
    source: str
    """Where the date came from: "git", "mtime", or "unknown"."""


UNKNOWN = FileDate("unknown", "unknown")


class DateLookup:
    """Dates files, asking git at most once per file.

    One instance per command run: an answer costs a subprocess, and a single
    answer can cite the same file several times.
    """

    def __init__(
        self,
        *,
        runner: Runner | None = None,
        timeout: float = GIT_TIMEOUT_SECONDS,
    ) -> None:
        self._runner: Runner = runner if runner is not None else subprocess.run
        self._timeout = timeout
        self._cache: dict[str, FileDate] = {}

    def date_for(self, path: Path | str) -> FileDate:
        key = str(path)
        if key not in self._cache:
            self._cache[key] = self._lookup(Path(path))
        return self._cache[key]

    def changed_since_indexed(self, path: Path | str, indexed_mtime: float) -> bool:
        """True when the file on disk is newer than the copy that was indexed.

        A file that has since been deleted is not "changed" - there is nothing to
        re-read, and the passage shown is still what was indexed.
        """
        current = self._mtime(Path(path))
        if current is None:
            return False
        return current - indexed_mtime > CHANGE_TOLERANCE_SECONDS

    def _lookup(self, path: Path) -> FileDate:
        committed = self._git_date(path)
        if committed is not None:
            return FileDate(committed, "git")
        mtime = self._mtime(path)
        if mtime is None:
            return UNKNOWN
        return FileDate(datetime.fromtimestamp(mtime).strftime(DATE_FORMAT), "mtime")

    def _git_date(self, path: Path) -> str | None:
        """The last commit date touching `path`, or None if git can't say."""
        try:
            result = self._runner(
                ["git", "-C", str(path.parent), "log", "-1", "--format=%cs", "--", str(path)],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            # Not installed, not executable, timed out - all just "no date".
            return None
        if result.returncode != 0:
            return None
        return self._valid_date(result.stdout.strip())

    @staticmethod
    def _valid_date(raw: str) -> str | None:
        """git's %cs is already YYYY-MM-DD; anything else is not trusted."""
        try:
            datetime.strptime(raw, DATE_FORMAT)
        except ValueError:
            return None
        return raw

    @staticmethod
    def _mtime(path: Path) -> float | None:
        try:
            return path.stat().st_mtime
        except OSError:
            return None
