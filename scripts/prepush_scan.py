"""Pre-push leak check over the whole tracked tree and every unpushed commit (see AGENTS.md).

Scans every tracked file at a commit (HEAD by default), not just the commit's diff, for
private terms, drive and home paths, email addresses and Google API keys. It then scans
every unpushed commit (origin/main..HEAD by default): each message, and the lines each
commit adds against its own parent - never the range's net diff, so a term added in one
commit and removed in a later one still fails. Private terms come only from the gitignored
config.local.toml: [scan] private_terms plus the private values its other tables already
hold (authors, excluded projects). Output names files, lines and term numbers - never a
term's text; a term inside a printed path is masked as [term #i]; a key is never printed.

Hits the owner has accepted are listed in [scan] accepted; each entry must match exactly
its `count` lines, so a new occurrence or a stale entry fails. In a commit, an added line
is accepted only if that commit's version of the file still matches the entry's count.
Accepted entries never cover a commit message or a key. Emails on reserved placeholder
domains are counted, not failed. Binary files (images, screenshots) are listed but not
read - check them by eye. A merge commit's message is scanned but not its diff: the
merged commits are scanned themselves, so only a conflict resolution goes unread.

The script never fetches: an unpushed commit cannot be on origin/main, so a stale
origin/main only means some pushed commits are scanned again.

Exit codes: 0 nothing new; 1 a new hit or a stale accepted entry; 2 config or git error
(including a missing upstream ref).

    python scripts/prepush_scan.py [--rev REV | --worktree] [--upstream REF] [--ignore-accepted]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from second_brain.config import AcceptedHit, ConfigError, read_scan_settings

_PATH_CHARS = r"[^\s'\"`)\]>,;]"
GENERIC: dict[str, re.Pattern[str]] = {
    "drive path": re.compile(rf"(?<![A-Za-z])[A-Za-z]:[\\/]{_PATH_CHARS}*"),
    # Not after ':' - a drive-letter path into a Users folder is already a drive path.
    "home path": re.compile(rf"(?<![\w:.])(?:/home/|/Users/){_PATH_CHARS}+"),
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    # Same shape as second_brain.git_history's Google key pattern - the project's real secret.
    "google api key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
}
# Never accepted, never printed.
_SECRET_KINDS = frozenset({"google api key"})
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
    commit: str = ""  # short hash for a commit hit; "" = the tree. file "" = the message


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
            if kind in _SECRET_KINDS:
                hits.append(Hit(file, number, kind, text, f"'{text[:4]}...' ({len(text)} chars)"))
                continue
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
        if hit.kind not in _SECRET_KINDS:
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


# --- unpushed commits -----------------------------------------------------------------

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass(frozen=True)
class CommitScan:
    short: str
    hits: list[Hit]  # message hits have file ""
    binary: list[str]
    merge: bool


def _diff_path(raw: str) -> str:
    path = raw.strip()
    if path.startswith('"') and path.endswith('"'):  # git quotes paths with odd characters
        path = path[1:-1]
    return path[2:] if path.startswith(("a/", "b/")) else path


def _scan_file_block(block: list[str], short: str, terms: Terms) -> tuple[list[Hit], str | None]:
    """Hits on one file's added lines (and its name, if new or renamed), and its path if binary."""
    header = block[0]
    path = header.rsplit(" b/", 1)[-1] if " b/" in header else ""
    new_name = False
    binary: str | None = None
    hits: list[Hit] = []
    line_no = 0
    in_hunks = False  # before the first hunk, "+++ " is a header, not an added line
    for raw in block[1:]:
        hunk = _HUNK.match(raw)
        if hunk:
            in_hunks = True
            line_no = int(hunk.group(1))
        elif not in_hunks:
            if raw.startswith("new file mode"):
                new_name = True
            elif raw.startswith(("rename to ", "copy to ")):
                new_name = True
                path = raw.split(" to ", 1)[1]
            elif raw.startswith("+++ ") and raw != "+++ /dev/null":
                path = _diff_path(raw[4:])
            elif raw.startswith("Binary files "):
                target = raw[len("Binary files ") :].rsplit(" and ", 1)[-1].removesuffix(" differ")
                if target != "/dev/null":
                    path = _diff_path(target)
                    binary = path
        elif raw.startswith("+"):
            hits.extend(replace(h, commit=short) for h in scan_line(path, line_no, raw[1:], terms))
            line_no += 1
    if new_name:
        hits.extend(
            Hit(path, 0, f"term #{index} in file name", path[start:end], commit=short)
            for start, end, index in terms.spans(path)
        )
    return hits, binary


def scan_commit(repo: Path, sha: str, terms: Terms) -> CommitScan:
    """One commit's message, and the lines it adds against its own parent."""
    short = sha[:7]
    parents = _git(repo, "rev-list", "--parents", "-n", "1", sha).decode().split()[1:]
    message = _git(repo, "log", "-1", "--format=%B", sha).decode("utf-8", errors="replace")
    hits = [
        replace(hit, commit=short)
        for number, line in enumerate(message.splitlines(), 1)
        for hit in scan_line("", number, line, terms)
    ]
    if len(parents) > 1:
        return CommitScan(short, hits, [], merge=True)

    diff = _git(
        repo, "-c", "core.quotePath=false", "diff-tree", "-p", "-M", "-U0",
        "--no-color", "--no-ext-diff", "--root", "--no-commit-id", sha,
    ).decode("utf-8", errors="replace")
    blocks: list[list[str]] = []
    for raw in diff.splitlines():
        if raw.startswith("diff --git "):
            blocks.append([raw])
        elif blocks:
            blocks[-1].append(raw)
    binary: list[str] = []
    for block in blocks:
        block_hits, binary_path = _scan_file_block(block, short, terms)
        hits.extend(block_hits)
        if binary_path is not None:
            binary.append(binary_path)
    return CommitScan(short, hits, binary, merge=False)


def classify_commit(
    repo: Path, sha: str, hits: Sequence[Hit], accepted: Sequence[AcceptedHit], terms: Terms
) -> tuple[list[Hit], list[tuple[Hit, str]]]:
    """(new, accepted with note). An added line is accepted only if that commit's version
    of the file still matches the entry's count; a message or a key never is."""
    entries = {(entry.file, entry.match.lower()): entry for entry in accepted}
    lines_at: dict[tuple[str, str], int] = {}
    new: list[Hit] = []
    accepted_hits: list[tuple[Hit, str]] = []
    for hit in hits:
        key = (hit.file, hit.text.lower())
        entry = entries.get(key)
        if hit.file == "" or hit.kind in _SECRET_KINDS or entry is None:
            new.append(hit)
            continue
        if key not in lines_at:
            blob = TrackedFile(hit.file, _git(repo, "show", f"{sha}:{hit.file}"))
            lines_at[key] = len(
                {h.line for h in scan([blob], terms)[0] if (h.file, h.text.lower()) == key}
            )
        if lines_at[key] == entry.count:
            accepted_hits.append((hit, entry.note))
        else:
            new.append(hit)
    return new, accepted_hits


def _location(hit: Hit, terms: Terms) -> str:
    if hit.commit and hit.file == "":
        return f"{hit.commit} message:{hit.line}"
    name = terms.mask(hit.file)
    if hit.commit:
        name = f"{hit.commit} +{name}"
    return name if hit.line == 0 else f"{name}:{hit.line}"


def _describe(hit: Hit, terms: Terms) -> str:
    return f"{_location(hit, terms)}  {hit.kind}" + (f" {hit.shown}" if hit.shown else "")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Whole-tracked-tree leak check before a push.")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--rev", default="HEAD", help="commit to scan (default HEAD)")
    where.add_argument("--worktree", action="store_true", help="scan tracked files on disk instead")
    parser.add_argument(
        "--upstream", default="origin/main", help="commits after this are unpushed (default origin/main)"
    )
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
    end = rev or "HEAD"  # the unpushed range ends here
    commit = b""
    try:
        if rev is not None:
            commit = _git(args.repo, "rev-parse", "--verify", "-q", f"{rev}^{{commit}}")
        files = tracked_files(args.repo, rev)
        end_sha = _git(args.repo, "rev-parse", "--verify", "-q", f"{end}^{{commit}}").decode().strip()
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip() if exc.stderr else ""
        print(f"git error: {' '.join(exc.cmd[3:])} failed. {detail}".rstrip(), file=sys.stderr)
        return 2
    try:
        upstream_sha = _git(
            args.repo, "rev-parse", "--verify", "-q", f"{args.upstream}^{{commit}}"
        ).decode().strip()
    except subprocess.CalledProcessError:
        # Fails closed: without it there is no telling which commits a push would publish.
        print(
            f"git error: upstream {args.upstream} not found - fetch it, or pass --upstream REF.",
            file=sys.stderr,
        )
        return 2
    counts = _git(args.repo, "rev-list", "--left-right", "--count", f"{end}...{args.upstream}")
    ahead, behind = (int(n) for n in counts.decode().split())
    unpushed = _git(args.repo, "rev-list", "--reverse", f"{args.upstream}..{end}").decode().split()

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

    if not unpushed:
        where_end = (
            f"{end} == {args.upstream} {upstream_sha[:7]}" if behind == 0
            else f"{end} is {behind} behind {args.upstream}"
        )
        print(f"commits: 0 unpushed ({where_end}) - nothing to scan")
    else:
        print(
            f"commits: {len(unpushed)} unpushed "
            f"({args.upstream} {upstream_sha[:7]}..{end} {end_sha[:7]})"
        )
        if behind:
            print(
                f"  ! diverged: {ahead} ahead, {behind} behind - a plain push will be refused; "
                "rebased commits get new hashes, so scan again after rebasing"
            )
    commit_new: list[Hit] = []
    commit_accepted: list[tuple[Hit, str]] = []
    commit_binary: list[tuple[str, str]] = []
    merges: list[str] = []
    for sha in unpushed:
        result = scan_commit(args.repo, sha, terms)
        placeholders += [hit for hit in result.hits if hit.kind == "placeholder email"]
        found = [hit for hit in result.hits if hit.kind != "placeholder email"]
        sha_new, sha_accepted = classify_commit(args.repo, sha, found, accepted, terms)
        commit_new += sha_new
        commit_accepted += sha_accepted
        commit_binary += [(result.short, path) for path in result.binary]
        if result.merge:
            merges.append(result.short)

    for hit in commit_new:
        print(f"  NEW       {_describe(hit, terms)}")
    for hit, note in commit_accepted:
        print(f"  ACCEPTED  {_describe(hit, terms)}" + (f"  ({note})" if note else ""))
    for short, path in commit_binary:
        print(f"  BINARY    {short} +{terms.mask(path)}  not scanned - check it by eye")
    for short in merges:
        print(f"  MERGE     {short}  diff not scanned (message scanned)")
    if commit_new:
        print(
            "  hint: a later commit does not unpublish an earlier one - "
            "rewrite the unpushed commits"
        )

    all_new = len(new) + len(commit_new)
    failed = bool(all_new or stale)
    print(
        f"new {all_new}, accepted {len(accepted_hits) + len(commit_accepted)}, "
        f"stale {len(stale)}, placeholder emails {len(placeholders)}, "
        f"binary not scanned {len(binary) + len(commit_binary)}, commits scanned {len(unpushed)}"
        f" -> {'FAIL' if failed else 'OK'}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
