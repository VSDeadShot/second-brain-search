"""Walk the scan root and decide which documents exist, and who they belong to.

`DiscoveredDoc.project` + `.rel_path` is the citation contract every later slice
prints, so it is deliberately the only thing this module is responsible for.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path

from .config import Config


@dataclass(frozen=True)
class DiscoveredDoc:
    path: Path
    project: str
    project_root: Path
    rel_path: str
    size_bytes: int
    mtime: float


def _matches_pattern(filename: str, patterns: tuple[str, ...]) -> bool:
    name = filename.lower()
    return any(fnmatch.fnmatchcase(name, pattern.lower()) for pattern in patterns)


def _under_included_dir(relative_dir: Path, include_dirs: frozenset[str]) -> bool:
    return any(part.lower() in include_dirs for part in relative_dir.parts)


def _nearest_git_root(start: Path, scan_root: Path, cache: dict[Path, Path | None]) -> Path | None:
    """Closest ancestor containing .git, searching no higher than the scan root."""
    chain: list[Path] = []
    current = start
    found: Path | None = None

    while True:
        if current in cache:
            found = cache[current]
            break
        chain.append(current)
        if (current / ".git").exists():
            found = current
            break
        if current == scan_root or current.parent == current:
            break
        current = current.parent

    for directory in chain:
        cache[directory] = found
    return found


def discover_documents(config: Config) -> list[DiscoveredDoc]:
    scan_root = config.scan_root.resolve()
    include_dirs = frozenset(d.lower() for d in config.include_dirs)
    exclude_dirs = frozenset(d.lower() for d in config.exclude_dirs)
    excluded = {(scan_root / p).resolve() for p in config.exclude_paths}

    git_cache: dict[Path, Path | None] = {}
    found: dict[Path, DiscoveredDoc] = {}

    for dirpath, dirnames, filenames in os.walk(scan_root, followlinks=False):
        current = Path(dirpath)

        # Prune CHILDREN only. The scan root is never a pruning candidate, so a root
        # that itself starts with "." still gets walked instead of silently yielding
        # nothing.
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and d.lower() not in exclude_dirs
            and (current / d).resolve() not in excluded
        ]

        try:
            relative_dir = current.relative_to(scan_root)
        except ValueError:  # pragma: no cover - os.walk stays under the root
            continue

        in_included_dir = _under_included_dir(relative_dir, include_dirs)

        for filename in filenames:
            is_match = _matches_pattern(filename, config.include_patterns) or (
                in_included_dir and filename.lower().endswith(".md")
            )
            if not is_match:
                continue

            path = (current / filename).resolve()
            if path in found:
                continue

            git_root = _nearest_git_root(current, scan_root, git_cache)
            if git_root is not None:
                project_root = git_root
            else:
                parts = path.relative_to(scan_root).parts
                project_root = scan_root / parts[0] if len(parts) > 1 else scan_root

            stat = path.stat()
            found[path] = DiscoveredDoc(
                path=path,
                project=project_root.name,
                project_root=project_root,
                rel_path=path.relative_to(project_root).as_posix(),
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
            )

    return sorted(found.values(), key=lambda d: (d.project.lower(), d.rel_path.lower()))
