"""Shared fixtures. Every tree is synthetic - no test reads the real Projects folder."""

import os
import subprocess
from pathlib import Path

import pytest
from dotenv import load_dotenv

# The project's documented config source is .env, so opt-in tests gated on
# GEMINI_API_KEY must see a key set there - not only a shell export. Runs at
# import time, before any skipif in a test module is evaluated. override=False
# keeps a real environment variable authoritative. Safe for hermeticity: every
# load_config() call in the suite passes an explicit env mapping.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


def make_file(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def make_git_dir(path: Path) -> Path:
    """Mark a directory as a repo root the way a real clone does."""
    (path / ".git").mkdir(parents=True, exist_ok=True)
    return path


# --- real git repos (git_history, and indexing with [git] enabled) ----------------

# Made-up authors: no real name or address appears in a test.
ME = ("Alice Example", "alice@example.com")
TEAMMATE = ("Bob Teammate", "bob@example.org")
GIT_DATE = "2026-07-02T10:00:00+05:30"


def git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=.no-hooks", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, **(env or {})},
        check=True,
    )
    return result.stdout.strip()


def init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.name", ME[0])
    git(path, "config", "user.email", ME[1])
    return path


def commit(
    repo: Path,
    message: str,
    files: dict[str, str | bytes] | None = None,
    *,
    author: tuple[str, str] = ME,
    date: str = GIT_DATE,
) -> str:
    for rel, content in (files or {}).items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    git(repo, "add", "-A")
    env = {
        "GIT_AUTHOR_NAME": author[0],
        "GIT_AUTHOR_EMAIL": author[1],
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_DATE": date,
    }
    git(repo, "commit", "-q", "--allow-empty", "-m", message, env=env)
    return git(repo, "rev-parse", "HEAD")


def clone_with_origin(clone: Path, *, work: Path) -> Path:
    """A clone of a bare origin holding one commit; upstream and origin live in `work`."""
    upstream = init_repo(work / f"{clone.name}-upstream")
    commit(upstream, "Initial commit", {"README.md": "hello\n"})
    bare = work / f"{clone.name}-origin.git"
    git(work, "clone", "-q", "--bare", str(upstream), str(bare))
    clone.parent.mkdir(parents=True, exist_ok=True)
    git(work, "clone", "-q", str(bare), str(clone))
    git(clone, "config", "user.name", ME[0])
    git(clone, "config", "user.email", ME[1])
    return clone


def make_git_chunk(
    *,
    project: str = "DSA Tracker",
    subject: str = "security: fix submitReview by verifying problem ownership",
    body: str = "The submitReview endpoint now checks the problem belongs to the caller.",
    author: tuple[str, str] = ME,
    date: str = GIT_DATE,
):
    """One commit chunk, built the way indexing builds it, without a real repo."""
    from datetime import datetime

    from second_brain.git_history import Commit, commit_chunks, commit_text

    commit_ = Commit(
        hash="b237453" + "0" * 33,
        parents=("a" * 40,),
        author_name=author[0],
        author_email=author[1],
        author_date=datetime.fromisoformat(date),
        subject=subject,
        body=body,
        files=("src/review.js",),
    )
    (chunk,) = commit_chunks(
        commit_, commit_text(commit_, excerpt=""), project=project, repo_path=Path("/p") / project
    )
    return chunk


ALPHA_README = (
    "# Alpha\n\nAlpha tracks practice problems and reviews, described at realistic length.\n"
)


@pytest.fixture
def git_corpus(tmp_path: Path) -> Path:
    """Three projects whose history is real git, for indexing with [git] enabled.

    root/
      Alpha/   clone of an origin (origin/main is the default branch)
               pushed: my fix + a teammate's commit; NOT pushed: one local commit
      Beta/    `git init`, no remote - its local HEAD is what gets read
      Gamma/   a bare `.git` directory that git cannot read - skipped with a reason
    """
    root = tmp_path / "root"
    alpha = clone_with_origin(root / "Alpha", work=tmp_path / "remotes")
    commit(alpha, "docs: describe Alpha", {"README.md": ALPHA_README})
    commit(
        alpha,
        "security: fix submitReview by verifying problem ownership",
        {"src/review.py": "def submit_review(user, problem):\n    check_owner(user, problem)\n"},
    )
    commit(
        alpha,
        "feat: teammate dashboard\n\nA teammate's change that must never be indexed.",
        {"src/dash.py": "DASH = 1\n"},
        author=TEAMMATE,
    )
    git(alpha, "push", "-q", "origin", "main")
    commit(alpha, "wip: local experiment never pushed", {"src/wip.py": "WIP = 1\n"})

    beta = init_repo(root / "Beta")
    commit(
        beta,
        "feat: beta scheduler\n\nSchedules jobs with a priority queue and retries failures "
        "with exponential backoff, so a flaky job cannot starve the rest.",
        {
            "README.md": "# Beta\n\nBeta is a job scheduler, and says so at some length here.\n",
            "src/sched.py": "QUEUE = []\n",
        },
    )

    gamma = make_git_dir(root / "Gamma")
    make_file(gamma / "README.md", "# Gamma\n\nGamma has docs but a .git that git cannot read.")
    return root


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A miniature of the real corpus, including every shape that tripped us up.

    root/
      Alpha/              (git root)
        README.md                          included
        CLAUDE.md                          included
        notes.md                           NOT matched by any pattern
        docs/superpowers/specs/design.md   included via include_dirs, nested deep
        node_modules/README.md             pruned (exclude_dirs)
        .pytest_cache/README.md            pruned (dot-directory)
      Parent/
        inner/            (git root)
          README.md                        project == "inner", not "Parent"
        orphan/           (no .git)
          README.md                        project falls back to "Parent"
      Lower/              (no .git)
        readme.md                          lowercase, must still match
        AI_Thing.md                        AI_*.md glob
      Skipped/
        sub/README.md                      pruned via exclude_paths
    """
    root = tmp_path / "root"

    alpha = make_git_dir(root / "Alpha")
    make_file(alpha / "README.md")
    make_file(alpha / "CLAUDE.md")
    make_file(alpha / "notes.md")
    make_file(alpha / "docs" / "superpowers" / "specs" / "design.md")
    make_file(alpha / "node_modules" / "README.md")
    make_file(alpha / ".pytest_cache" / "README.md")

    inner = make_git_dir(root / "Parent" / "inner")
    make_file(inner / "README.md")
    make_file(root / "Parent" / "orphan" / "README.md")

    make_file(root / "Lower" / "readme.md")
    make_file(root / "Lower" / "AI_Thing.md")

    make_file(root / "Skipped" / "sub" / "README.md")

    return root


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Two small git projects with realistic-length docs: Alpha (2 files), Beta (1)."""
    root = tmp_path / "root"
    alpha = make_git_dir(root / "Alpha")
    make_file(
        alpha / "README.md",
        "# Alpha\n\nAlpha does a thing worth describing at realistic length.\n\n"
        "## Detail\n\nMore detail about how the thing actually works in practice.",
    )
    make_file(
        alpha / "CLAUDE.md",
        "# Notes\n\nSome notes about Alpha, long enough to clear the size floor.",
    )
    beta = make_git_dir(root / "Beta")
    make_file(
        beta / "README.md",
        "# Beta\n\nBeta is different from Alpha, and says so at some length.",
    )
    return root


@pytest.fixture
def knowledge(tmp_path: Path) -> Path:
    """Three projects with distinctive vocabularies, for retrieval ranking tests."""
    root = tmp_path / "knowledge"
    make_file(
        make_git_dir(root / "Macro Tracker") / "CLAUDE.md",
        "# Photos\n\nMeal photo compression happens client side with a canvas, "
        "shrinking each jpeg photo before upload so compression keeps payloads small.",
    )
    make_file(
        make_git_dir(root / "Watch Tracker") / "EXPLAINER.md",
        "# Caching\n\nThe caching layer keeps show metadata in redis with a ttl, "
        "so repeated caching lookups skip the upstream api entirely.",
    )
    make_file(
        make_git_dir(root / "RDBMS") / "README.md",
        "# Storage\n\nThe storage engine writes pages to disk through a buffer pool "
        "and a write ahead log so storage survives a crash.",
    )
    return root


@pytest.fixture
def doc_factory(tmp_path: Path):
    """Build a DiscoveredDoc without needing a real discovery walk."""
    from second_brain.discovery import DiscoveredDoc

    def make(rel_path: str = "README.md", project: str = "Proj") -> DiscoveredDoc:
        root = tmp_path / project
        path = root / rel_path
        make_file(path)
        return DiscoveredDoc(
            path=path,
            project=project,
            project_root=root,
            rel_path=rel_path,
            size_bytes=path.stat().st_size,
            mtime=path.stat().st_mtime,
        )

    return make


# --- MCP server: a corpus with more chunks than the default k --------------------

# Over 240 characters, so a search that returned the CLI's snippet instead of the
# full passage would be caught.
LONG_CACHING_PASSAGE = (
    "# Caching\n\nThe caching layer keeps show metadata in redis with a ttl, so "
    "repeated caching lookups skip the upstream api entirely. Cache keys carry the "
    "api version, so a schema change never serves a stale shape, and every caching "
    "miss is logged with its latency so slow upstream calls show up in the caching "
    "dashboard before users notice them."
)


@pytest.fixture
def search_corpus(tmp_path: Path) -> Path:
    """Seven docs across three projects. "Hidden" is the one the server must hide,
    and it is the best match for caching questions - a leak would rank first."""
    root = tmp_path / "search_corpus"
    alpha = make_git_dir(root / "Alpha")
    make_file(alpha / "README.md", LONG_CACHING_PASSAGE)
    make_file(alpha / "CLAUDE.md", "# Storage\n\nPages are written to disk through a buffer pool and a write ahead log.")
    make_file(alpha / "docs" / "auth.md", "# Auth\n\nSessions are signed cookies, rotated on every login and checked on each request.")
    beta = make_git_dir(root / "Beta")
    make_file(beta / "README.md", "# Photos\n\nMeal photos are compressed client side with a canvas before upload.")
    make_file(beta / "CLAUDE.md", "# Caching\n\nA small caching layer memoises the nutrition lookups for an hour.")
    make_file(beta / "docs" / "deploy.md", "# Deploy\n\nThe app deploys from main through a container build and a health check.")
    hidden = make_git_dir(root / "Hidden")
    make_file(hidden / "README.md", "# Caching\n\nHidden caching notes: caching caching caching with a private cache warmer.")
    return root


def index_corpus(corpus: Path, index_dir: Path, embedder) -> "ChunkStore":
    """Index `corpus` the way `sbs index` does, recording `embedder`'s vector space."""
    from second_brain.config import DEFAULT_EXCLUDE_DIRS, DEFAULT_INCLUDE_PATTERNS, Config
    from second_brain.pipeline import collect_chunks, store_chunks
    from second_brain.store import ChunkStore

    config = Config(
        scan_root=corpus,
        gemini_api_key=None,
        include_patterns=DEFAULT_INCLUDE_PATTERNS,
        include_dirs=("docs",),
        exclude_dirs=DEFAULT_EXCLUDE_DIRS,
        exclude_paths=(),
    )
    store = ChunkStore(index_dir)
    store_chunks(collect_chunks(config), embedder, store, rebuild=True)
    return store


def stub_gemini_embedder(models=None):
    """The real GeminiEmbedder over a stub client - its vector space is Gemini's, so a
    store built with it accepts queries from the server's own embedder."""
    from second_brain.embedding import DEFAULT_DIMENSIONS, GeminiEmbedder

    from fakes import StubClient, StubModels

    models = models if models is not None else StubModels(dimensions=DEFAULT_DIMENSIONS)
    return GeminiEmbedder(
        client=StubClient(models),
        items_per_minute=None,
        tokens_per_minute=None,
        sleep=lambda _seconds: None,
    )
