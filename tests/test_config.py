"""Config loading: defaults -> config.toml -> config.local.toml -> environment."""

from pathlib import Path

import pytest

from second_brain.config import ConfigError, load_config


def write_toml(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_scan_root_comes_from_the_environment(tmp_path: Path) -> None:
    scan = tmp_path / "projects"
    scan.mkdir()

    cfg = load_config(project_root=tmp_path, env={"SBS_SCAN_ROOT": str(scan)})

    assert cfg.scan_root == scan.resolve()


def test_missing_scan_root_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="SBS_SCAN_ROOT"):
        load_config(project_root=tmp_path, env={})


def test_scan_root_that_is_not_a_directory_is_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "nope"

    with pytest.raises(ConfigError, match="nope"):
        load_config(project_root=tmp_path, env={"SBS_SCAN_ROOT": str(missing)})


def test_gemini_key_is_optional_for_discovery(tmp_path: Path) -> None:
    scan = tmp_path / "projects"
    scan.mkdir()

    cfg = load_config(project_root=tmp_path, env={"SBS_SCAN_ROOT": str(scan)})

    assert cfg.gemini_api_key is None


def test_gemini_key_is_read_when_present(tmp_path: Path) -> None:
    scan = tmp_path / "projects"
    scan.mkdir()

    cfg = load_config(
        project_root=tmp_path,
        env={"SBS_SCAN_ROOT": str(scan), "GEMINI_API_KEY": "secret-value"},
    )

    assert cfg.gemini_api_key == "secret-value"


def test_defaults_apply_with_no_config_file(tmp_path: Path) -> None:
    scan = tmp_path / "projects"
    scan.mkdir()

    cfg = load_config(project_root=tmp_path, env={"SBS_SCAN_ROOT": str(scan)})

    assert "README.md" in cfg.include_patterns
    assert "AI_*.md" in cfg.include_patterns
    assert "node_modules" in cfg.exclude_dirs
    assert cfg.include_dirs == ("docs",)


def test_config_toml_overrides_defaults(tmp_path: Path) -> None:
    scan = tmp_path / "projects"
    scan.mkdir()
    write_toml(
        tmp_path / "config.toml",
        '[discovery]\ninclude_patterns = ["ONLY.md"]\nexclude_paths = ["SIH/driftless-int"]\n',
    )

    cfg = load_config(project_root=tmp_path, env={"SBS_SCAN_ROOT": str(scan)})

    assert cfg.include_patterns == ("ONLY.md",)
    assert cfg.exclude_paths == ("SIH/driftless-int",)
    # Untouched keys keep their defaults.
    assert "node_modules" in cfg.exclude_dirs


def test_config_local_toml_wins_over_config_toml(tmp_path: Path) -> None:
    scan = tmp_path / "projects"
    scan.mkdir()
    write_toml(tmp_path / "config.toml", '[discovery]\ninclude_dirs = ["docs"]\n')
    write_toml(tmp_path / "config.local.toml", '[discovery]\ninclude_dirs = ["notes"]\n')

    cfg = load_config(project_root=tmp_path, env={"SBS_SCAN_ROOT": str(scan)})

    assert cfg.include_dirs == ("notes",)


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    """A typo in config.toml should fail loudly, not be silently ignored."""
    scan = tmp_path / "projects"
    scan.mkdir()
    write_toml(tmp_path / "config.toml", '[discovery]\ninclude_pattern = ["typo.md"]\n')

    with pytest.raises(ConfigError, match="include_pattern"):
        load_config(project_root=tmp_path, env={"SBS_SCAN_ROOT": str(scan)})
