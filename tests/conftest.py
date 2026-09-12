"""Shared fixtures. Every tree is synthetic - no test reads the real Projects folder."""

from pathlib import Path

import pytest


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
