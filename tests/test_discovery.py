"""Discovery behaviour. Each test names the rule it defends."""

from pathlib import Path

import pytest

from second_brain.config import Config, DEFAULT_EXCLUDE_DIRS, DEFAULT_INCLUDE_PATTERNS
from second_brain.discovery import discover_documents

from conftest import make_file, make_git_dir


def config_for(root: Path, **overrides) -> Config:
    base = dict(
        scan_root=root,
        gemini_api_key=None,
        include_patterns=DEFAULT_INCLUDE_PATTERNS,
        include_dirs=("docs",),
        exclude_dirs=DEFAULT_EXCLUDE_DIRS,
        exclude_paths=("Skipped",),
    )
    base.update(overrides)
    return Config(**base)


def rel_paths(docs) -> set[str]:
    return {d.path.name for d in docs}


def by_name(docs, name: str):
    return next(d for d in docs if d.path.name == name)


def test_finds_named_patterns_in_a_project(tree: Path) -> None:
    docs = discover_documents(config_for(tree))
    names = [d.path.name for d in docs]
    assert "README.md" in names
    assert "CLAUDE.md" in names


def test_unmatched_markdown_is_left_out(tree: Path) -> None:
    docs = discover_documents(config_for(tree))
    assert "notes.md" not in rel_paths(docs)


def test_dot_directories_are_pruned(tree: Path) -> None:
    """.pytest_cache/README.md matches a pattern but must never be indexed."""
    docs = discover_documents(config_for(tree))
    assert not any(".pytest_cache" in str(d.path) for d in docs)


def test_scan_root_may_itself_be_a_dot_directory(tmp_path: Path) -> None:
    """Regression: pruning every dot-dir must not prune the root and return nothing."""
    root = tmp_path / ".hidden"
    make_file(make_git_dir(root / "Proj") / "README.md")

    docs = discover_documents(config_for(root))

    assert len(docs) == 1
    assert docs[0].project == "Proj"


def test_exclude_dirs_are_pruned(tree: Path) -> None:
    docs = discover_documents(config_for(tree))
    assert not any("node_modules" in str(d.path) for d in docs)


def test_exclude_paths_prune_a_whole_subtree(tree: Path) -> None:
    docs = discover_documents(config_for(tree))
    assert not any("Skipped" in str(d.path) for d in docs)


def test_nearest_git_ancestor_names_the_project(tree: Path) -> None:
    """Parent/inner has its own .git, so the project is 'inner', not 'Parent'."""
    doc = by_name([d for d in discover_documents(config_for(tree)) if "inner" in str(d.path)], "README.md")
    assert doc.project == "inner"


def test_project_falls_back_to_top_level_folder(tree: Path) -> None:
    """Parent/orphan has no .git anywhere below Parent, so it cites as 'Parent'."""
    doc = by_name([d for d in discover_documents(config_for(tree)) if "orphan" in str(d.path)], "README.md")
    assert doc.project == "Parent"
    assert doc.rel_path == "orphan/README.md"


def test_docs_directory_matches_recursively(tree: Path) -> None:
    """A literal docs/specs/*.md glob misses docs/superpowers/specs/ - this must not."""
    docs = discover_documents(config_for(tree))
    assert "design.md" in rel_paths(docs)


def test_pattern_matching_is_case_insensitive(tree: Path) -> None:
    docs = discover_documents(config_for(tree))
    assert "readme.md" in rel_paths(docs)


def test_ai_prefix_glob_matches(tree: Path) -> None:
    docs = discover_documents(config_for(tree))
    assert "AI_Thing.md" in rel_paths(docs)


def test_rel_path_is_posix_and_relative_to_project_root(tree: Path) -> None:
    doc = by_name(discover_documents(config_for(tree)), "design.md")
    assert doc.rel_path == "docs/superpowers/specs/design.md"
    assert doc.project == "Alpha"


def test_order_is_stable_across_runs(tree: Path) -> None:
    """Slice 2 chunks in this order, so it must not shift between runs."""
    first = [(d.project, d.rel_path) for d in discover_documents(config_for(tree))]
    second = [(d.project, d.rel_path) for d in discover_documents(config_for(tree))]
    assert first == second


def test_order_is_case_insensitive_by_project_then_path(tree: Path) -> None:
    """README.md and readme.md group together rather than splitting on case."""
    keys = [(d.project, d.rel_path) for d in discover_documents(config_for(tree))]
    assert keys == sorted(keys, key=lambda k: (k[0].lower(), k[1].lower()))


def test_metadata_is_populated(tree: Path) -> None:
    doc = by_name(discover_documents(config_for(tree)), "CLAUDE.md")
    assert doc.size_bytes > 0
    assert doc.mtime > 0
    assert doc.path.is_absolute()


def test_no_duplicate_paths(tree: Path) -> None:
    docs = discover_documents(config_for(tree))
    paths = [d.path for d in docs]
    assert len(paths) == len(set(paths))


def test_bare_spec_and_brief_names_match(tmp_path: Path) -> None:
    """A hyphenless SPEC.md is as much a spec as project-name-SPEC.md is."""
    root = tmp_path / "root"
    proj = make_git_dir(root / "Proj")
    make_file(proj / "SPEC.md")
    make_file(proj / "BRIEF.md")
    make_file(proj / "other-SPEC.md")

    names = {d.path.name for d in discover_documents(config_for(root))}

    assert names == {"SPEC.md", "BRIEF.md", "other-SPEC.md"}


def test_exclude_paths_can_name_a_single_file(tmp_path: Path) -> None:
    """Not every exclusion is a directory - the tool's own changelog is one file."""
    root = tmp_path / "root"
    proj = make_git_dir(root / "Proj")
    make_file(proj / "README.md")
    make_file(proj / "CLAUDE_SUMMARY.md")

    docs = discover_documents(config_for(root, exclude_paths=("Proj/CLAUDE_SUMMARY.md",)))

    assert {d.path.name for d in docs} == {"README.md"}


def test_excluding_a_file_leaves_its_siblings_alone(tmp_path: Path) -> None:
    """Excluding one file must not prune the directory holding it."""
    root = tmp_path / "root"
    proj = make_git_dir(root / "Proj")
    make_file(proj / "docs" / "keep.md")
    make_file(proj / "docs" / "drop.md")

    docs = discover_documents(config_for(root, exclude_paths=("Proj/docs/drop.md",)))

    assert {d.path.name for d in docs} == {"keep.md"}
