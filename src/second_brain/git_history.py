"""Read a repo's commit history and turn the owner's commits into chunks.

Docs record what a project is; commits often hold the only record of how
something was fixed. This module reads a repo's history through the git CLI (a
list of arguments, no shell, a timeout) and produces `Chunk`s shaped like a doc's,
so the rest of the pipeline can treat both sources alike. `repo_status` picks the
ref: the remote's default branch as last fetched, or local HEAD with no remote.

A commit becomes one text: subject, body, the files it changed, and for a thin
commit (no body, or a short message) a small diff excerpt, because a thin
commit's message alone rarely says enough. The excerpt never includes lockfiles,
images, binaries, env files or keys, and it drops any line that looks like a
secret. The commit text never names its repo; `embed_text` does, since questions
usually name the project.

Nothing here touches the index; pipeline.collect_chunks calls in when [git] is on.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath

from .chunking import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    DEFAULT_OVERLAP,
    Chunk,
    _split_with_overlap,
)

DEFAULT_TIMEOUT = 30.0
DEFAULT_THIN_MESSAGE_CHARS = 200
DEFAULT_DIFF_EXCERPT_CHARS = 800
MAX_FILES_LISTED = 30

# Pinned per call so the owner's own git config cannot change the output shape.
_GIT_CONFIG = (
    "-c", "core.quotePath=false",
    "-c", "log.showSignature=false",
    "-c", "log.showRoot=true",
    "-c", "color.ui=never",
)  # fmt: skip

# Variables that would point git at some other repo than the one asked for.
_REPO_ENV_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR")

_FIELD = "\x1f"
_RECORD = "\x1e"
# Hash, parents, author name, email, strict ISO author date, subject, body - then
# --name-only appends the changed paths after the last separator.
_LOG_FORMAT = "%x1e%H%x1f%P%x1f%an%x1f%ae%x1f%aI%x1f%s%x1f%b%x1f"
_LOG_FIELDS = 8

_HASH_RE = re.compile(r"[0-9a-f]{40,64}")

# "chore: bump version to 1.1.0", "chore(release): bump version ...", or a bare
# "1.0.8". A commit that bumps a version alongside real work ("docs: ... and bump
# version to 1.0.4") is kept.
_VERSION_BUMP_RE = re.compile(
    r"^(?:chore(?:\([^)]*\))?:\s*bump version\b.*|v?\d+(?:\.\d+)+)$", re.IGNORECASE
)

_LOCKFILES = frozenset(
    {
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lock",
        "bun.lockb",
        "poetry.lock",
        "pipfile.lock",
        "uv.lock",
        "cargo.lock",
        "gemfile.lock",
        "composer.lock",
        "go.sum",
    }
)
_IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".svg", ".tif", ".tiff", ".avif"}
)

# Any diff line matching one of these is dropped whole - never redacted in place,
# since a partial match could leave the rest of a secret behind.
_SECRET_PATTERNS = (
    # Google API key.
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    # PEM private key header: BEGIN RSA PRIVATE KEY, BEGIN OPENSSH PRIVATE KEY, ...
    re.compile(r"BEGIN [A-Z0-9 ]*KEY"),
    # key=, secret=, password=, token= (or ':') with a non-empty value.
    re.compile(
        r"(?:key|secret|password|passwd|token)\w*[\"']?\s*[:=]\s*[\"']?[^\s\"',;]+",
        re.IGNORECASE,
    ),
    # Credentials embedded in a URL: scheme://user:pass@host
    re.compile(r"\b[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^/\s@]+@", re.IGNORECASE),
    # bcrypt hash.
    re.compile(r"\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}"),
)


class GitError(Exception):
    """A git command failed; the caller skips that repo and reports why."""


@dataclass(frozen=True)
class Commit:
    hash: str
    parents: tuple[str, ...]
    author_name: str
    author_email: str
    author_date: datetime
    subject: str
    body: str
    files: tuple[str, ...]

    @property
    def short(self) -> str:
        return self.hash[:7]

    @property
    def is_root(self) -> bool:
        return not self.parents

    @property
    def is_merge(self) -> bool:
        return len(self.parents) > 1

    @property
    def message(self) -> str:
        return f"{self.subject}\n\n{self.body}" if self.body else self.subject


@dataclass(frozen=True)
class CommitSelection:
    kept: list[Commit]
    merges: int = 0
    version_bumps: int = 0
    other_authors: int = 0


@dataclass(frozen=True)
class RepoHistory:
    chunks: list[Chunk]
    commits: int
    """Commits read from HEAD."""
    kept: int
    merges: int
    version_bumps: int
    other_authors: int
    with_excerpt: int
    """Thin commits whose text gained a diff excerpt."""


@dataclass(frozen=True)
class RepoStatus:
    ref: str
    """What history is read from: the remote default branch (e.g. "origin/main"),
    or "HEAD" for a repo with no remote or no resolvable default branch."""
    branch: str | None
    """Checked-out branch, or None for a detached HEAD."""
    default_branch: str | None
    has_remote: bool
    ahead: int = 0
    """Commits on local HEAD that `ref` lacks - they will not be indexed."""
    behind: int = 0
    """Commits on `ref` that local HEAD lacks."""
    flags: list[str] = field(default_factory=list)


# --- running git -------------------------------------------------------------------


def _run_git(repo: Path, args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[bytes]:
    env = {k: v for k, v in os.environ.items() if k not in _REPO_ENV_VARS}
    # Stops git climbing out of `repo` into some enclosing repository.
    env["GIT_CEILING_DIRECTORIES"] = str(Path(repo).resolve().parent)
    try:
        return subprocess.run(
            ["git", *_GIT_CONFIG, *args],
            cwd=repo,
            env=env,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {args[0]} timed out after {timeout:g}s") from exc
    except OSError as exc:
        raise GitError(f"could not run git in {repo}: {exc}") from exc


def _git(repo: Path, *args: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    result = _run_git(repo, args, timeout)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        reason = stderr.splitlines()[0] if stderr else f"exit status {result.returncode}"
        raise GitError(f"git {args[0]} failed: {reason}")
    return result.stdout.decode("utf-8", errors="replace")


def _git_optional(repo: Path, *args: str, timeout: float = DEFAULT_TIMEOUT) -> str | None:
    """Output, or None when git exits non-zero (e.g. a ref that does not exist)."""
    result = _run_git(repo, args, timeout)
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="replace").strip()


# --- reading -----------------------------------------------------------------------


def read_commits(
    repo: Path, *, ref: str = "HEAD", timeout: float = DEFAULT_TIMEOUT
) -> list[Commit]:
    """Every commit reachable from `ref`, newest first."""
    if not ref or ref.startswith("-"):
        raise ValueError(f"not a ref: {ref!r}")
    out = _git(repo, "log", ref, f"--format={_LOG_FORMAT}", "--name-only", timeout=timeout)

    commits: list[Commit] = []
    for record in out.split(_RECORD)[1:]:
        parts = record.split(_FIELD)
        if len(parts) != _LOG_FIELDS:
            raise GitError(f"unexpected git log output in {repo}")
        full_hash, parents, name, email, date, subject, body, paths = parts
        commits.append(
            Commit(
                hash=full_hash,
                parents=tuple(parents.split()),
                author_name=name,
                author_email=email,
                author_date=datetime.fromisoformat(date),
                subject=subject.strip(),
                body=body.strip(),
                files=tuple(line for line in paths.splitlines() if line.strip()),
            )
        )
    return commits


def read_diff(repo: Path, commit_hash: str, *, timeout: float = DEFAULT_TIMEOUT) -> str:
    """The commit's changes with no context lines (-U0)."""
    if not _HASH_RE.fullmatch(commit_hash):
        raise ValueError(f"not a full commit hash: {commit_hash!r}")
    return _git(
        repo,
        "show", "--format=", "-U0", "--no-color", "--no-ext-diff", "--no-textconv",
        commit_hash,
        timeout=timeout,
    )  # fmt: skip


# --- filtering ---------------------------------------------------------------------


def is_version_bump(subject: str) -> bool:
    return bool(_VERSION_BUMP_RE.match(subject.strip()))


def _normalise_name(name: str) -> str:
    return " ".join(name.split()).casefold()


def filter_commits(
    commits: Iterable[Commit],
    *,
    author_emails: Iterable[str],
    author_names: Iterable[str],
) -> CommitSelection:
    """Drop merges, version bumps and other people's commits, counting each.

    A commit is kept when its author matches EITHER list (case-insensitive).
    Empty lists match nobody.
    """
    emails = {e.strip().casefold() for e in author_emails if e.strip()}
    names = {_normalise_name(n) for n in author_names if n.strip()}

    kept: list[Commit] = []
    merges = bumps = others = 0
    for commit in commits:
        if commit.is_merge:
            merges += 1
        elif is_version_bump(commit.subject):
            bumps += 1
        elif (
            commit.author_email.strip().casefold() not in emails
            and _normalise_name(commit.author_name) not in names
        ):
            others += 1
        else:
            kept.append(commit)

    return CommitSelection(kept=kept, merges=merges, version_bumps=bumps, other_authors=others)


def is_thin(commit: Commit, *, thin_message_chars: int = DEFAULT_THIN_MESSAGE_CHARS) -> bool:
    """No body, or a message too short to say much on its own."""
    return not commit.body or len(commit.message) < thin_message_chars


# --- diff excerpt ------------------------------------------------------------------


def _diff_path(raw: str) -> str:
    # Git appends a tab to ---/+++ paths containing spaces, and quotes unusual ones.
    path = raw.rstrip().strip('"')
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def _excluded_from_excerpt(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    return (
        name in _LOCKFILES
        or PurePosixPath(name).suffix in _IMAGE_SUFFIXES
        or name.startswith(".env")
        or name.endswith(".pem")
    )


def _looks_secret(line: str) -> bool:
    return any(pattern.search(line) for pattern in _SECRET_PATTERNS)


def diff_excerpt(diff: str, *, max_chars: int = DEFAULT_DIFF_EXCERPT_CHARS) -> str:
    """Changed lines and hunk headers, minus excluded files and secret-looking lines."""
    kept: list[str] = []
    skip_file = False
    in_header = False

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            in_header = True
            skip_file = _excluded_from_excerpt(_diff_path(line.rsplit(" b/", 1)[-1]))
            continue
        if in_header:
            # Header lines run until the first hunk; inside a hunk, "--- x" is
            # just a removed line that happened to start with "--".
            if line.startswith(("--- ", "+++ ")):
                path = _diff_path(line[4:])
                if path != "/dev/null":
                    skip_file = skip_file or _excluded_from_excerpt(path)
                continue
            if not line.startswith("@@"):
                continue
            in_header = False

        if skip_file or not line.startswith(("+", "-", "@@")):
            continue
        if _looks_secret(line):
            continue
        kept.append(line)

    return "\n".join(kept)[:max_chars].rstrip()


# --- text and chunks ---------------------------------------------------------------


def commit_text(commit: Commit, *, excerpt: str) -> str:
    parts = [commit.subject]
    if commit.body:
        parts.append(commit.body)
    if commit.files:
        listed = list(commit.files[:MAX_FILES_LISTED])
        extra = len(commit.files) - MAX_FILES_LISTED
        if extra > 0:
            listed.append(f"... and {extra} more")
        parts.append("Files changed:\n" + "\n".join(listed))
    if excerpt:
        parts.append("Diff excerpt:\n" + excerpt)
    return "\n\n".join(parts)


def commit_chunks(
    commit: Commit,
    text: str,
    *,
    project: str,
    repo_path: Path,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """One chunk per piece of the commit text; the same size floor as docs."""
    header = f"{project} · commit {commit.short} · {commit.author_date.date().isoformat()}"
    pieces = [p for p in _split_with_overlap(text, max_chars, overlap) if len(p) >= min_chars]

    chunks: list[Chunk] = []
    for index, piece in enumerate(pieces):
        # Later pieces lose the subject line; put it back for the embedding.
        body = piece if piece.startswith(commit.subject) else f"{commit.subject}\n\n{piece}"
        embed_text = f"{header}\n{body}"
        identity = f"{project}|git:{commit.hash}|{index}".encode()
        chunks.append(
            Chunk(
                chunk_id=hashlib.sha256(identity).hexdigest(),
                project=project,
                rel_path=f"commit {commit.short}",
                path=str(repo_path),
                heading_path=commit.subject,
                chunk_index=index,
                text=piece,
                embed_text=embed_text,
                content_hash=hashlib.sha256(embed_text.encode("utf-8")).hexdigest(),
                mtime=commit.author_date.timestamp(),
                source="git",
                commit=commit.hash,
                author=commit.author_name,
                author_email=commit.author_email,
                author_date=commit.author_date.isoformat(),
            )
        )
    return chunks


def collect_repo_history(
    repo: Path,
    *,
    project: str,
    author_emails: Iterable[str],
    author_names: Iterable[str],
    ref: str = "HEAD",
    thin_message_chars: int = DEFAULT_THIN_MESSAGE_CHARS,
    diff_excerpt_chars: int = DEFAULT_DIFF_EXCERPT_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    min_chars: int = DEFAULT_MIN_CHARS,
    timeout: float = DEFAULT_TIMEOUT,
) -> RepoHistory:
    """Read, filter and chunk one repo's history at `ref`. Raises GitError if git fails."""
    commits = read_commits(repo, ref=ref, timeout=timeout)
    selection = filter_commits(commits, author_emails=author_emails, author_names=author_names)

    chunks: list[Chunk] = []
    with_excerpt = 0
    for commit in selection.kept:
        excerpt = ""
        # A root commit's diff is the whole initial tree - not an excerpt of a change.
        if not commit.is_root and is_thin(commit, thin_message_chars=thin_message_chars):
            excerpt = diff_excerpt(
                read_diff(repo, commit.hash, timeout=timeout), max_chars=diff_excerpt_chars
            )
            if excerpt:
                with_excerpt += 1
        chunks.extend(
            commit_chunks(
                commit,
                commit_text(commit, excerpt=excerpt),
                project=project,
                repo_path=repo,
                max_chars=max_chars,
                overlap=overlap,
                min_chars=min_chars,
            )
        )

    return RepoHistory(
        chunks=chunks,
        commits=len(commits),
        kept=len(selection.kept),
        merges=selection.merges,
        version_bumps=selection.version_bumps,
        other_authors=selection.other_authors,
        with_excerpt=with_excerpt,
    )


# --- repo status (for the dry run) -------------------------------------------------


def _remote_default_branch(repo: Path, remote: str, timeout: float) -> str | None:
    """The remote's default branch as last fetched: <remote>/HEAD, else main, else master."""
    head_ref = _git_optional(
        repo, "symbolic-ref", "--short", "-q", f"refs/remotes/{remote}/HEAD", timeout=timeout
    )
    if head_ref:
        return head_ref.removeprefix(f"{remote}/")
    for candidate in ("main", "master"):
        ref = f"refs/remotes/{remote}/{candidate}"
        if _git_optional(repo, "rev-parse", "--verify", "-q", ref, timeout=timeout):
            return candidate
    return None


def repo_status(repo: Path, *, timeout: float = DEFAULT_TIMEOUT) -> RepoStatus:
    """Which ref to read history from, and how the local checkout compares to it.

    A repo with a remote is read at the remote's default branch, so the index
    holds what was pushed, whatever happens to be checked out. Remote-tracking
    refs are used as of the last fetch - this never touches the network.
    """
    remotes = _git(repo, "remote", timeout=timeout).split()
    branch = _git_optional(repo, "symbolic-ref", "--short", "-q", "HEAD", timeout=timeout) or None

    flags: list[str] = []
    if branch is None:
        flags.append("detached HEAD")

    if not remotes:
        return RepoStatus(
            ref="HEAD",
            branch=branch,
            default_branch=None,
            has_remote=False,
            flags=flags + ["no remote - reading local HEAD"],
        )

    remote = "origin" if "origin" in remotes else remotes[0]
    default_branch = _remote_default_branch(repo, remote, timeout)
    if default_branch is None:
        return RepoStatus(
            ref="HEAD",
            branch=branch,
            default_branch=None,
            has_remote=True,
            flags=flags + ["default branch unknown - reading local HEAD"],
        )

    ref = f"{remote}/{default_branch}"
    counts = _git(repo, "rev-list", "--left-right", "--count", f"HEAD...{ref}", timeout=timeout)
    ahead, behind = (int(n) for n in counts.split())

    if branch is not None and branch != default_branch:
        flags.append(f"checked out {branch}, not {default_branch}")
    if ahead:
        flags.append(f"local is {ahead} ahead of {ref}")
    if behind:
        flags.append(f"local is {behind} behind {ref}")

    return RepoStatus(
        ref=ref,
        branch=branch,
        default_branch=default_branch,
        has_remote=True,
        ahead=ahead,
        behind=behind,
        flags=flags,
    )
