"""Configuration loading.

Precedence, lowest to highest: code defaults -> config.toml -> config.local.toml
-> environment. Secrets and machine-local paths come from the environment (a .env
file); list-shaped settings live in TOML where they stay diffable and commentable.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import dotenv_values


class ConfigError(Exception):
    """Configuration is missing, malformed, or points at something that isn't there."""


DEFAULT_INCLUDE_PATTERNS: tuple[str, ...] = (
    "README.md",
    "CLAUDE_SUMMARY.md",
    "PROJECT_STATUS.md",
    "CLAUDE.md",
    "AGENTS*.md",
    "DECISIONS.md",
    "ROADMAP.md",
    "HANDOVER.md",
    "EXPLAINER.md",
    "*BRIEF.md",
    "*SPEC.md",
    "AI_*.md",
)

DEFAULT_INCLUDE_DIRS: tuple[str, ...] = ("docs",)

# Dot-directories are pruned as a class, so .git/.venv/.pytest_cache need no entry.
DEFAULT_EXCLUDE_DIRS: tuple[str, ...] = (
    "node_modules",
    "venv",
    "build",
    "dist",
    "__pycache__",
    "target",
)

# Directories or single files, relative to the scan root.
DEFAULT_EXCLUDE_PATHS: tuple[str, ...] = (
    "SIH/driftless-int",
    "Second Brain/CLAUDE_SUMMARY.md",
)

_LIST_FIELDS = ("include_patterns", "include_dirs", "exclude_dirs", "exclude_paths")


@dataclass(frozen=True)
class Config:
    scan_root: Path
    gemini_api_key: str | None
    include_patterns: tuple[str, ...]
    include_dirs: tuple[str, ...]
    exclude_dirs: tuple[str, ...]
    exclude_paths: tuple[str, ...]


def _repo_root() -> Path:
    # src/second_brain/config.py -> src/second_brain -> src -> repo root
    return Path(__file__).resolve().parents[2]


def _read_discovery_table(path: Path) -> dict[str, tuple[str, ...]]:
    if not path.is_file():
        return {}

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path.name} is not valid TOML: {exc}") from exc

    table = raw.get("discovery", {})
    if not isinstance(table, dict):
        raise ConfigError(f"{path.name}: [discovery] must be a table")

    unknown = sorted(set(table) - set(_LIST_FIELDS))
    if unknown:
        raise ConfigError(
            f"{path.name}: unknown key(s) under [discovery]: {', '.join(unknown)}. "
            f"Valid keys are: {', '.join(_LIST_FIELDS)}"
        )

    values: dict[str, tuple[str, ...]] = {}
    for key, value in table.items():
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"{path.name}: [discovery].{key} must be a list of strings")
        values[key] = tuple(value)
    return values


def _resolve_scan_root(env: Mapping[str, str]) -> Path:
    raw = (env.get("SBS_SCAN_ROOT") or "").strip()
    if not raw:
        raise ConfigError(
            "SBS_SCAN_ROOT is not set. Copy .env.example to .env and point it at the "
            "parent folder holding your project repos."
        )

    root = Path(raw).expanduser()
    if not root.is_dir():
        raise ConfigError(f"SBS_SCAN_ROOT points at {root}, which is not an existing directory.")
    return root.resolve()


def load_config(
    project_root: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Build a Config.

    `project_root` is where config.toml / config.local.toml / .env live; it defaults
    to the repo root. Passing `env` explicitly bypasses the .env file entirely, which
    is what keeps the tests hermetic.
    """
    root = (project_root or _repo_root()).resolve()

    if env is None:
        # Real environment wins over the .env file.
        env = {**dotenv_values(root / ".env"), **os.environ}

    config = Config(
        scan_root=_resolve_scan_root(env),
        gemini_api_key=(env.get("GEMINI_API_KEY") or None),
        include_patterns=DEFAULT_INCLUDE_PATTERNS,
        include_dirs=DEFAULT_INCLUDE_DIRS,
        exclude_dirs=DEFAULT_EXCLUDE_DIRS,
        exclude_paths=DEFAULT_EXCLUDE_PATHS,
    )

    for name in ("config.toml", "config.local.toml"):
        overrides = _read_discovery_table(root / name)
        if overrides:
            config = replace(config, **overrides)

    return config
