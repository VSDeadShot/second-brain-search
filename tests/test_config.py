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


# --- the generation model -------------------------------------------------------


def test_the_generation_model_defaults_to_the_one_with_the_headroom(tmp_path: Path) -> None:
    scan = tmp_path / "projects"
    scan.mkdir()

    cfg = load_config(project_root=tmp_path, env={"SBS_SCAN_ROOT": str(scan)})

    assert cfg.generation_model == "gemini-3.5-flash-lite"


def test_the_generation_model_can_be_changed_from_the_environment(tmp_path: Path) -> None:
    """Swapping to the documented fallback is one line in .env, not a code change."""
    scan = tmp_path / "projects"
    scan.mkdir()

    cfg = load_config(
        project_root=tmp_path,
        env={"SBS_SCAN_ROOT": str(scan), "SBS_GENERATION_MODEL": "gemini-3.6-flash"},
    )

    assert cfg.generation_model == "gemini-3.6-flash"


def test_a_blank_generation_model_falls_back_to_the_default(tmp_path: Path) -> None:
    scan = tmp_path / "projects"
    scan.mkdir()

    cfg = load_config(
        project_root=tmp_path, env={"SBS_SCAN_ROOT": str(scan), "SBS_GENERATION_MODEL": "   "}
    )

    assert cfg.generation_model == "gemini-3.5-flash-lite"


# --- [git] ---------------------------------------------------------------------


def _scan(tmp_path: Path) -> dict[str, str]:
    scan = tmp_path / "projects"
    scan.mkdir()
    return {"SBS_SCAN_ROOT": str(scan)}


def test_git_history_is_off_by_default_in_code(tmp_path: Path) -> None:
    cfg = load_config(project_root=tmp_path, env=_scan(tmp_path))

    assert cfg.git_enabled is False
    assert cfg.git_author_emails == ()
    assert cfg.git_author_names == ()
    assert cfg.git_diff_excerpt_chars == 800
    assert cfg.git_thin_message_chars == 200


def test_git_settings_are_read_from_config_toml(tmp_path: Path) -> None:
    write_toml(
        tmp_path / "config.toml",
        "[git]\nenabled = true\nauthor_emails = []\nauthor_names = []\n"
        "diff_excerpt_chars = 500\nthin_message_chars = 150\n",
    )

    cfg = load_config(project_root=tmp_path, env=_scan(tmp_path))

    assert cfg.git_enabled is True
    assert cfg.git_diff_excerpt_chars == 500
    assert cfg.git_thin_message_chars == 150


def test_config_local_toml_supplies_the_authors_and_keeps_the_rest(tmp_path: Path) -> None:
    write_toml(
        tmp_path / "config.toml",
        "[git]\nenabled = true\nauthor_emails = []\nauthor_names = []\n",
    )
    write_toml(
        tmp_path / "config.local.toml",
        '[git]\nauthor_emails = ["me@example.com"]\nauthor_names = ["Me Myself"]\n',
    )

    cfg = load_config(project_root=tmp_path, env=_scan(tmp_path))

    assert cfg.git_enabled is True  # from config.toml, untouched by the local file
    assert cfg.git_author_emails == ("me@example.com",)
    assert cfg.git_author_names == ("Me Myself",)


def test_a_local_discovery_table_does_not_reset_git_settings(tmp_path: Path) -> None:
    write_toml(tmp_path / "config.toml", "[git]\nenabled = true\n")
    write_toml(tmp_path / "config.local.toml", '[discovery]\ninclude_dirs = ["notes"]\n')

    cfg = load_config(project_root=tmp_path, env=_scan(tmp_path))

    assert cfg.git_enabled is True
    assert cfg.include_dirs == ("notes",)


def test_unknown_git_key_is_rejected(tmp_path: Path) -> None:
    write_toml(tmp_path / "config.toml", '[git]\nauthor_email = ["typo@example.com"]\n')

    with pytest.raises(ConfigError, match="author_email"):
        load_config(project_root=tmp_path, env=_scan(tmp_path))


@pytest.mark.parametrize(
    "body",
    [
        'enabled = "yes"',
        'author_emails = "me@example.com"',
        "author_names = [1, 2]",
        "diff_excerpt_chars = 0",
        "thin_message_chars = -5",
        'thin_message_chars = "200"',
        "diff_excerpt_chars = true",
    ],
)
def test_badly_typed_git_values_are_rejected(tmp_path: Path, body: str) -> None:
    write_toml(tmp_path / "config.toml", f"[git]\n{body}\n")

    with pytest.raises(ConfigError, match=r"\[git\]"):
        load_config(project_root=tmp_path, env=_scan(tmp_path))


def test_committed_config_enables_git_with_no_author_values() -> None:
    """The real author lists belong only in the gitignored config.local.toml."""
    import tomllib

    committed = Path(__file__).resolve().parents[1] / "config.toml"
    table = tomllib.loads(committed.read_text(encoding="utf-8"))["git"]

    assert table["enabled"] is True
    assert table["author_emails"] == []
    assert table["author_names"] == []


# --- [mcp] - what the MCP server hides -------------------------------------------


def test_exclude_projects_is_unset_when_no_file_sets_it(tmp_path: Path) -> None:
    """Unset is not the same as empty: the server refuses to run on unset."""
    cfg = load_config(project_root=tmp_path, env=_scan(tmp_path))

    assert cfg.mcp_exclude_projects is None


def test_an_explicit_empty_exclude_list_is_kept_distinct_from_unset(tmp_path: Path) -> None:
    write_toml(tmp_path / "config.local.toml", "[mcp]\nexclude_projects = []\n")

    cfg = load_config(project_root=tmp_path, env=_scan(tmp_path))

    assert cfg.mcp_exclude_projects == ()


def test_config_local_toml_supplies_the_excluded_projects(tmp_path: Path) -> None:
    write_toml(tmp_path / "config.toml", "[mcp]\n")
    write_toml(
        tmp_path / "config.local.toml",
        '[mcp]\nexclude_projects = ["Private One", "Private Two"]\n',
    )

    cfg = load_config(project_root=tmp_path, env=_scan(tmp_path))

    assert cfg.mcp_exclude_projects == ("Private One", "Private Two")


def test_unknown_mcp_key_is_rejected(tmp_path: Path) -> None:
    write_toml(tmp_path / "config.local.toml", '[mcp]\nexclude_project = ["Private One"]\n')

    with pytest.raises(ConfigError, match="exclude_project"):
        load_config(project_root=tmp_path, env=_scan(tmp_path))


@pytest.mark.parametrize(
    "body",
    ['exclude_projects = "Private One"', "exclude_projects = [1]", "exclude_projects = true"],
)
def test_badly_typed_mcp_values_are_rejected(tmp_path: Path, body: str) -> None:
    write_toml(tmp_path / "config.local.toml", f"[mcp]\n{body}\n")

    with pytest.raises(ConfigError, match=r"\[mcp\]"):
        load_config(project_root=tmp_path, env=_scan(tmp_path))


def test_committed_config_leaves_exclude_projects_unset() -> None:
    """The real names belong only in the gitignored config.local.toml, and leaving the
    key unset here is what makes a missing local file fail closed."""
    import tomllib

    committed = Path(__file__).resolve().parents[1] / "config.toml"
    raw = tomllib.loads(committed.read_text(encoding="utf-8"))

    assert "mcp" in raw
    assert "exclude_projects" not in raw["mcp"]


def test_max_searches_defaults_to_50(tmp_path: Path) -> None:
    cfg = load_config(project_root=tmp_path, env=_scan(tmp_path))

    assert cfg.mcp_max_searches == 50


def test_max_searches_can_be_set_locally_without_touching_the_exclusions(tmp_path: Path) -> None:
    write_toml(tmp_path / "config.toml", "[mcp]\nmax_searches = 50\n")
    write_toml(tmp_path / "config.local.toml", '[mcp]\nmax_searches = 10\nexclude_projects = ["P"]\n')

    cfg = load_config(project_root=tmp_path, env=_scan(tmp_path))

    assert cfg.mcp_max_searches == 10
    assert cfg.mcp_exclude_projects == ("P",)


@pytest.mark.parametrize("value", ["0", "-1", '"50"', "true", "2.5"])
def test_badly_typed_max_searches_is_rejected(tmp_path: Path, value: str) -> None:
    write_toml(tmp_path / "config.toml", f"[mcp]\nmax_searches = {value}\n")

    with pytest.raises(ConfigError, match=r"\[mcp\]\.max_searches"):
        load_config(project_root=tmp_path, env=_scan(tmp_path))


def test_committed_config_caps_a_session_at_50_searches() -> None:
    import tomllib

    committed = Path(__file__).resolve().parents[1] / "config.toml"
    raw = tomllib.loads(committed.read_text(encoding="utf-8"))

    assert raw["mcp"]["max_searches"] == 50
