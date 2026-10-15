"""Pre-push leak check over the whole tracked tree (see AGENTS.md).

Scans every tracked file at a commit (HEAD by default), not just the commit's diff, for
private terms, drive and home paths, and email addresses. Private terms come only from
the gitignored config.local.toml: [scan] private_terms plus the private values its other
tables already hold (authors, excluded projects). Output names files, lines and term
numbers - never a term's text; a term inside a printed path is masked as [term #i].

Hits the owner has accepted are listed in [scan] accepted; each entry must match exactly
its `count` lines, so a new occurrence or a stale entry fails. Emails on reserved
placeholder domains are counted, not failed. Binary files (images, screenshots) are
listed but not read - check them by eye.

Exit codes: 0 nothing new; 1 a new hit or a stale accepted entry; 2 config or git error.

    python scripts/prepush_scan.py [--rev REV | --worktree] [--ignore-accepted]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from second_brain.config import AcceptedHit, ConfigError, read_scan_settings

_PATH_CHARS = r"[^\s'\"`)\]>,;]"
GENERIC: dict[str, re.Pattern[str]] = {
    "drive path": re.compile(rf"(?<![A-Za-z])[A-Za-z]:[\\/]{_PATH_CHARS}*"),
    # Not after ':' - a drive-letter path into a Users folder is already a drive path.
    "home path": re.compile(rf"(?<![\w:.])(?:/home/|/Users/){_PATH_CHARS}+"),
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
}
# Reserved for documentation and testing (RFC 2606, 6761, 6762) - never a real mailbox.
_PLACEHOLDER_EMAIL = re.compile(
    r"@(?:[\w-]+\.)*(?:example\.(?:com|org|net)|[\w-]+\.(?:invalid|test|local|localhost|example))$",
    re.IGNORECASE,
)
# Git's own heuristic: a NUL byte in the first 8000 bytes means binary.
_BINARY_SNIFF = 8000


@dataclass(frozen=True)
class Hit:
    file: str
    line: int  # 0 = the file's name
    kind: str
    text: str  # what matched - compared with accepted entries, never printed for a term
    shown: str = ""  # printed after the kind; terms inside it are masked


@dataclass(frozen=True)
class TrackedFile:
    path: str
    content: bytes


class Terms:
    """Case-insensitive term matching; where terms overlap, the longest match wins."""

    def __init__(self, terms: Sequence[str]) -> None:
        self.lowered = [t.lower() for t in terms]

    def spans(self, text: str) -> list[tuple[int, int, int]]:
        low = text.lower()
        found = []
        for index, term in enumerate(self.lowered):
            start = low.find(term)
            while start != -1:
                found.append((start, start + len(term), index))
                start = low.find(term, start + 1)
        found.sort(key=lambda s: (s[0], s[0] - s[1]))
        kept: list[tuple[int, int, int]] = []
        for span in found:
            if not any(k[0] <= span[0] and span[1] <= k[1] for k in kept):
                kept.append(span)
        return kept

    def mask(self, text: str) -> str:
        out, last = [], 0
        for start, end, index in sorted(self.spans(text)):
            if start < last:
                continue
            out.append(text[last:start] + f"[term #{index}]")
            last = end
        return "".join(out) + text[last:]


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout


def tracked_files(repo: Path, rev: str | None) -> list[TrackedFile]:
    """Every tracked blob at `rev`, or in the working tree when `rev` is None."""
    if rev is None:
        names = [n for n in _git(repo, "ls-files", "-z").decode("utf-8").split("\0") if n]
        return [TrackedFile(n, (repo / n).read_bytes()) for n in names if (repo / n).is_file()]

    files = []
    for entry in _git(repo, "ls-tree", "-r", "-z", rev).decode("utf-8").split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        _mode, kind, blob = meta.split()
        if kind == "blob":  # submodules are commits, not files
            files.append(TrackedFile(path, _git(repo, "cat-file", "blob", blob)))
    return files


def scan_line(file: str, number: int, line: str, terms: Terms) -> list[Hit]:
    hits = [
        Hit(file, number, f"term #{index}", line[start:end])
        for start, end, index in terms.spans(line)
    ]
    for kind, pattern in GENERIC.items():
        for match in pattern.finditer(line):
            text = match.group(0)
            if kind == "email" and _PLACEHOLDER_EMAIL.search(text):
                kind_shown = "placeholder email"
            else:
                kind_shown = kind
            hits.append(Hit(file, number, kind_shown, text, repr(terms.mask(text))))
    return hits


def scan(files: Sequence[TrackedFile], terms: Terms) -> tuple[list[Hit], list[str]]:
    """All hits, and the binary files that were not read."""
    hits: list[Hit] = []
    binary: list[str] = []
    for f in files:
        for start, end, index in terms.spans(f.path):
            hits.append(Hit(f.path, 0, f"term #{index} in file name", f.path[start:end]))
        if b"\0" in f.content[:_BINARY_SNIFF]:
            binary.append(f.path)
            continue
        text = f.content.decode("utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            hits.extend(scan_line(f.path, number, line, terms))
    return hits, binary


def classify(
    hits: Sequence[Hit], accepted: Sequence[AcceptedHit]
) -> tuple[list[Hit], list[tuple[Hit, str]], list[tuple[int, AcceptedHit, int]]]:
    """(new, accepted with note, stale entries as (position, entry, lines found))."""
    by_key: dict[tuple[str, str], list[Hit]] = {}
    for hit in hits:
        by_key.setdefault((hit.file, hit.text.lower()), []).append(hit)

    accepted_hits: list[tuple[Hit, str]] = []
    covered: set[int] = set()
    stale: list[tuple[int, AcceptedHit, int]] = []
    for position, entry in enumerate(accepted, 1):
        group = by_key.get((entry.file, entry.match.lower()), [])
        lines = len({hit.line for hit in group})
        if lines == entry.count:
            accepted_hits.extend((hit, entry.note) for hit in group)
            covered.update(id(hit) for hit in group)
        elif lines < entry.count:
            stale.append((position, entry, lines))
        # More lines than accepted: every one stays new.
    new = [hit for hit in hits if id(hit) not in covered]
    return new, accepted_hits, stale


def _location(hit: Hit, terms: Terms) -> str:
    name = terms.mask(hit.file)
    return name if hit.line == 0 else f"{name}:{hit.line}"


def _describe(hit: Hit, terms: Terms) -> str:
    return f"{_location(hit, terms)}  {hit.kind}" + (f" {hit.shown}" if hit.shown else "")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Whole-tracked-tree leak check before a push.")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--rev", default="HEAD", help="commit to scan (default HEAD)")
    where.add_argument("--worktree", action="store_true", help="scan tracked files on disk instead")
    parser.add_argument(
        "--ignore-accepted", action="store_true", help="report accepted hits as new (positive control)"
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    try:
        settings = read_scan_settings(args.repo)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    all_terms = list(dict.fromkeys([*settings.private_terms, *settings.config_terms]))
    terms = Terms(all_terms)

    rev = None if args.worktree else args.rev
    commit = b""
    try:
        if rev is not None:
            commit = _git(args.repo, "rev-parse", "--verify", "-q", f"{rev}^{{commit}}")
        files = tracked_files(args.repo, rev)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip() if exc.stderr else ""
        print(f"git error: {' '.join(exc.cmd[3:])} failed. {detail}".rstrip(), file=sys.stderr)
        return 2

    target = "in the working tree" if rev is None else f"at {rev} ({commit.decode().strip()[:7]})"
    print(f"prepush scan: {len(files)} tracked files {target}; {len(all_terms)} private terms")

    hits, binary = scan(files, terms)
    placeholders = [hit for hit in hits if hit.kind == "placeholder email"]
    hits = [hit for hit in hits if hit.kind != "placeholder email"]
    accepted = () if args.ignore_accepted else settings.accepted
    new, accepted_hits, stale = classify(hits, accepted)

    for hit in new:
        print(f"  NEW       {_describe(hit, terms)}")
    for hit, note in accepted_hits:
        print(f"  ACCEPTED  {_describe(hit, terms)}" + (f"  ({note})" if note else ""))
    for position, entry, lines in stale:
        print(
            f"  STALE     accepted entry {position} ({terms.mask(entry.file)}): "
            f"expects {entry.count} line(s), found {lines}"
        )
    for path in binary:
        print(f"  BINARY    {terms.mask(path)}  not scanned - check it by eye")

    failed = bool(new or stale)
    print(
        f"new {len(new)}, accepted {len(accepted_hits)}, stale {len(stale)}, "
        f"placeholder emails {len(placeholders)}, binary not scanned {len(binary)}"
        f" -> {'FAIL' if failed else 'OK'}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
