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
from typing import Any, cast

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

# [mcp] key -> (Config field, expected kind).
_MCP_FIELDS: dict[str, tuple[str, str]] = {
    "exclude_projects": ("mcp_exclude_projects", "strings"),
    "max_searches": ("mcp_max_searches", "positive int"),
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
    # Projects never indexed: their docs are skipped at discovery, so their history is
    # never read and none of their text is sent to Gemini. Set only in the gitignored
    # config.local.toml - the names are private. None means unset; () skips nothing.
    discovery_exclude_projects: tuple[str, ...] | None = None
    # Searches one MCP server process (one client session) may send to Gemini.
    mcp_max_searches: int = 50


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


def _typed_value(table: str, key: str, value: object, kind: str, filename: str) -> object:
    if kind == "bool" and isinstance(value, bool):
        return value
    if kind == "strings" and _is_string_list(value):
        return tuple(value)  # type: ignore[arg-type]
    # bool is an int subclass, so `true` has to be refused explicitly here.
    is_int = isinstance(value, int) and not isinstance(value, bool)
    if kind == "positive int" and is_int and value > 0:  # type: ignore[operator]
        return value
    raise ConfigError(f"{filename}: [{table}].{key} must be {_EXPECTED[kind]}")


def _parse_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path.name} is not valid TOML: {exc}") from exc


def _read_overrides(path: Path) -> dict[str, Any]:
    """Config fields set by one TOML file - only the keys it actually contains."""
    if not path.is_file():
        return {}

    raw = _parse_toml(path)
    values: dict[str, Any] = {}
    discovery_keys = (*_LIST_FIELDS, "exclude_projects")
    for key, value in _table(raw, "discovery", discovery_keys, path.name).items():
        if not _is_string_list(value):
            raise ConfigError(f"{path.name}: [discovery].{key} must be a list of strings")
        field_name = "discovery_exclude_projects" if key == "exclude_projects" else key
        values[field_name] = tuple(value)

    for key, value in _table(raw, "git", _GIT_FIELDS, path.name).items():
        field_name, kind = _GIT_FIELDS[key]
        values[field_name] = _typed_value("git", key, value, kind, path.name)

    for key, value in _table(raw, "mcp", _MCP_FIELDS, path.name).items():
        field_name, kind = _MCP_FIELDS[key]
        values[field_name] = _typed_value("mcp", key, value, kind, path.name)
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


# --- [scan] - read only by scripts/prepush_scan.py ----------------------------------
#
# Deliberately not part of Config: load_config never reads [scan], so the leak check's
# settings cannot change what is indexed, discovered or served over MCP.

_SCAN_KEYS = ("private_terms", "accepted")
_ACCEPTED_KEYS = ("file", "match", "count", "note")
_ACCEPTED_REQUIRED = ("file", "match", "count")

# Config fields whose values are private, so the scan looks for them too.
_PRIVATE_CONFIG_FIELDS = (
    "git_author_emails",
    "git_author_names",
    "discovery_exclude_projects",
    "mcp_exclude_projects",
)


@dataclass(frozen=True)
class AcceptedHit:
    """A scan hit the owner has accepted: `count` lines of `file` may carry `match`."""

    file: str
    match: str
    count: int
    note: str = ""


@dataclass(frozen=True)
class ScanSettings:
    private_terms: tuple[str, ...]
    # The private values the other tables already hold (authors, excluded projects).
    config_terms: tuple[str, ...]
    accepted: tuple[AcceptedHit, ...] = ()


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _accepted_entries(value: object, filename: str) -> tuple[AcceptedHit, ...]:
    # Refusals give the entry's position, never its text - the match may be a private term.
    if not isinstance(value, list):
        raise ConfigError(f"{filename}: [scan].accepted must be a list of tables")

    entries: list[AcceptedHit] = []
    seen: set[tuple[str, str]] = set()
    for position, entry in enumerate(value, 1):
        where = f"{filename}: [scan].accepted entry {position}"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where} must be a table")
        unknown = sorted(set(entry) - set(_ACCEPTED_KEYS))
        if unknown:
            raise ConfigError(
                f"{where}: unknown key(s): {', '.join(unknown)}. "
                f"Valid keys are: {', '.join(_ACCEPTED_KEYS)}"
            )
        missing = [key for key in _ACCEPTED_REQUIRED if key not in entry]
        if missing:
            raise ConfigError(f"{where} is missing: {', '.join(missing)}")
        for key in ("file", "match"):
            if not _non_empty_string(entry[key]):
                raise ConfigError(f"{where}: {key} must be a non-empty string")
        count = entry["count"]
        # bool is an int subclass, so `true` has to be refused explicitly here.
        if not (isinstance(count, int) and not isinstance(count, bool) and count > 0):
            raise ConfigError(f"{where}: count must be a positive integer")
        note = entry.get("note", "")
        if not isinstance(note, str):
            raise ConfigError(f"{where}: note must be a string")

        # Matching is case-insensitive, so two entries differing only in case collide.
        key = (entry["file"], entry["match"].lower())
        if key in seen:
            raise ConfigError(f"{where} repeats an earlier entry's file and match")
        seen.add(key)
        entries.append(AcceptedHit(file=entry["file"], match=entry["match"], count=count, note=note))
    return tuple(entries)


def read_scan_settings(project_root: Path | None = None) -> ScanSettings:
    """The pre-push scan's settings, from config.local.toml only.

    Fails closed: a missing file, [scan] table or private_terms key is refused rather
    than scanning with no terms. `private_terms = []` is allowed.
    """
    root = (project_root or _repo_root()).resolve()
    path = root / "config.local.toml"
    if not path.is_file():
        raise ConfigError(
            "config.local.toml not found. The pre-push scan reads [scan] private_terms from it."
        )

    raw = _parse_toml(path)
    if "scan" not in raw:
        raise ConfigError(f"{path.name}: [scan] is not set. Add [scan] private_terms = [...] to it.")
    table = _table(raw, "scan", _SCAN_KEYS, path.name)
    if "private_terms" not in table:
        raise ConfigError(f"{path.name}: [scan].private_terms is not set ([] scans for none).")
    terms = cast(
        tuple[str, ...],
        _typed_value("scan", "private_terms", table["private_terms"], "strings", path.name),
    )
    for position, term in enumerate(terms, 1):
        if not term.strip():
            # A blank term would match every line.
            raise ConfigError(f"{path.name}: [scan].private_terms entry {position} is blank")

    overrides = _read_overrides(path)
    config_terms: list[str] = []
    for field_name in _PRIVATE_CONFIG_FIELDS:
        config_terms.extend(v for v in overrides.get(field_name) or () if v.strip())

    return ScanSettings(
        private_terms=terms,
        config_terms=tuple(dict.fromkeys(config_terms)),
        accepted=_accepted_entries(table.get("accepted", []), path.name),
    )
