"""Configuration loading.

Precedence, lowest to highest: code defaults -> config.toml -> config.local.toml
-> environment. Secrets and machine-local paths come from the environment (a .env
file); list-shaped settings live in TOML where they stay diffable and commentable.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

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

# Generation models. Lite is the default for its free-tier headroom: 500 requests
# a day and 15 a minute on this project, against 20 a day and 5 a minute for the
# 3.x Flash models. SBS_GENERATION_MODEL switches to the fallback if Lite's
# answers turn out weak.
DEFAULT_GENERATION_MODEL = "gemini-3.5-flash-lite"
FALLBACK_GENERATION_MODEL = "gemini-3.6-flash"

_LIST_FIELDS = ("include_patterns", "include_dirs", "exclude_dirs", "exclude_paths")

# [git] key -> (Config field, expected kind).
_GIT_FIELDS: dict[str, tuple[str, str]] = {
    "enabled": ("git_enabled", "bool"),
    "author_emails": ("git_author_emails", "strings"),
    "author_names": ("git_author_names", "strings"),
    "diff_excerpt_chars": ("git_diff_excerpt_chars", "positive int"),
    "thin_message_chars": ("git_thin_message_chars", "positive int"),
}


@dataclass(frozen=True)
class Config:
    scan_root: Path
    gemini_api_key: str | None
    include_patterns: tuple[str, ...]
    include_dirs: tuple[str, ...]
    exclude_dirs: tuple[str, ...]
    exclude_paths: tuple[str, ...]
    generation_model: str = DEFAULT_GENERATION_MODEL
    # Off unless config.toml turns it on, so a bare Config indexes docs only.
    git_enabled: bool = False
    # Only commits by these authors are indexed - a match on EITHER list. The
    # real values live in the gitignored config.local.toml.
    git_author_emails: tuple[str, ...] = ()
    git_author_names: tuple[str, ...] = ()
    git_diff_excerpt_chars: int = 800
    git_thin_message_chars: int = 200
    # Projects the MCP server hides. None (unset) is not () (hide nothing): the
    # server refuses to run on None, so a missing config.local.toml fails closed.
    mcp_exclude_projects: tuple[str, ...] | None = None


def _repo_root() -> Path:
    # src/second_brain/config.py -> src/second_brain -> src -> repo root
    return Path(__file__).resolve().parents[2]


def _table(raw: dict[str, Any], name: str, valid: Iterable[str], filename: str) -> dict[str, Any]:
    table = raw.get(name, {})
    if not isinstance(table, dict):
        raise ConfigError(f"{filename}: [{name}] must be a table")
    valid = tuple(valid)
    unknown = sorted(set(table) - set(valid))
    if unknown:
        raise ConfigError(
            f"{filename}: unknown key(s) under [{name}]: {', '.join(unknown)}. "
            f"Valid keys are: {', '.join(valid)}"
        )
    return table


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


_EXPECTED = {
    "bool": "true or false",
    "strings": "a list of strings",
    "positive int": "a positive integer",
}


def _git_value(key: str, value: object, kind: str, filename: str) -> object:
    if kind == "bool" and isinstance(value, bool):
        return value
    if kind == "strings" and _is_string_list(value):
        return tuple(value)  # type: ignore[arg-type]
    # bool is an int subclass, so `true` has to be refused explicitly here.
    is_int = isinstance(value, int) and not isinstance(value, bool)
    if kind == "positive int" and is_int and value > 0:  # type: ignore[operator]
        return value
    raise ConfigError(f"{filename}: [git].{key} must be {_EXPECTED[kind]}")


def _read_overrides(path: Path) -> dict[str, Any]:
    """Config fields set by one TOML file - only the keys it actually contains."""
    if not path.is_file():
        return {}

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path.name} is not valid TOML: {exc}") from exc

    values: dict[str, Any] = {}
    for key, value in _table(raw, "discovery", _LIST_FIELDS, path.name).items():
        if not _is_string_list(value):
            raise ConfigError(f"{path.name}: [discovery].{key} must be a list of strings")
        values[key] = tuple(value)

    for key, value in _table(raw, "git", _GIT_FIELDS, path.name).items():
        field_name, kind = _GIT_FIELDS[key]
        values[field_name] = _git_value(key, value, kind, path.name)

    for key, value in _table(raw, "mcp", ("exclude_projects",), path.name).items():
        if not _is_string_list(value):
            raise ConfigError(f"{path.name}: [mcp].{key} must be a list of strings")
        values["mcp_exclude_projects"] = tuple(value)
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
        generation_model=(env.get("SBS_GENERATION_MODEL") or "").strip() or DEFAULT_GENERATION_MODEL,
    )

    # Key by key: a local file that sets only the author lists leaves config.toml's
    # other [git] values (and its [discovery] table) in place.
    for name in ("config.toml", "config.local.toml"):
        overrides = _read_overrides(root / name)
        if overrides:
            config = replace(config, **overrides)

    return config
