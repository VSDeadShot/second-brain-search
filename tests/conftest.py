"""Shared fixtures. Every tree is synthetic - no test reads the real Projects folder."""

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
