"""The scan-only [scan] table: read by the pre-push scan, never by load_config."""

from pathlib import Path

import pytest

from second_brain.config import AcceptedHit, ConfigError, read_scan_settings


def write_local(root: Path, body: str) -> None:
    (root / "config.local.toml").write_text(body, encoding="utf-8")


def test_reads_private_terms_and_accepted_entries(tmp_path: Path) -> None:
    write_local(
        tmp_path,
        '[scan]\nprivate_terms = ["Alicia", "alicia99"]\n'
        "accepted = [\n"
        '  { file = "SPEC.md", match = "Alicia", count = 1, note = "kept" },\n'
        '  { file = "tests/test_x.py", match = "docs/a.md", count = 2 },\n'
        "]\n",
    )

    settings = read_scan_settings(tmp_path)

    assert settings.private_terms == ("Alicia", "alicia99")
    assert settings.accepted == (
        AcceptedHit(file="SPEC.md", match="Alicia", count=1, note="kept"),
        AcceptedHit(file="tests/test_x.py", match="docs/a.md", count=2, note=""),
    )


def test_collects_the_private_values_from_the_other_tables(tmp_path: Path) -> None:
    write_local(
        tmp_path,
        '[scan]\nprivate_terms = ["Alicia"]\n'
        '[git]\nauthor_emails = ["alicia@example.com"]\nauthor_names = ["Alicia Q"]\n'
        '[discovery]\nexclude_projects = ["Hidden One"]\n'
        '[mcp]\nexclude_projects = ["Hidden One", "Hidden Two"]\n',
    )

    settings = read_scan_settings(tmp_path)

    assert settings.config_terms == ("alicia@example.com", "Alicia Q", "Hidden One", "Hidden Two")


def test_other_tables_are_optional(tmp_path: Path) -> None:
    write_local(tmp_path, '[scan]\nprivate_terms = ["Alicia"]\n')

    settings = read_scan_settings(tmp_path)

    assert settings.config_terms == ()
    assert settings.accepted == ()


def test_an_empty_private_terms_list_is_allowed(tmp_path: Path) -> None:
    write_local(tmp_path, "[scan]\nprivate_terms = []\n")

    assert read_scan_settings(tmp_path).private_terms == ()


def test_reads_only_config_local_toml(tmp_path: Path) -> None:
    """Private terms in the committed config.toml would be public; they are never read."""
    (tmp_path / "config.toml").write_text('[scan]\nprivate_terms = ["Alicia"]\n', encoding="utf-8")
    write_local(tmp_path, '[scan]\nprivate_terms = ["Bertil"]\n')

    assert read_scan_settings(tmp_path).private_terms == ("Bertil",)


# --- fails closed -----------------------------------------------------------------


def test_a_missing_config_local_toml_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config.local.toml"):
        read_scan_settings(tmp_path)


def test_a_missing_scan_table_is_refused(tmp_path: Path) -> None:
    write_local(tmp_path, '[git]\nauthor_emails = ["alicia@example.com"]\n')

    with pytest.raises(ConfigError, match=r"\[scan\]"):
        read_scan_settings(tmp_path)


def test_a_missing_private_terms_key_is_refused(tmp_path: Path) -> None:
    write_local(tmp_path, "[scan]\naccepted = []\n")

    with pytest.raises(ConfigError, match="private_terms"):
        read_scan_settings(tmp_path)


def test_invalid_toml_is_refused(tmp_path: Path) -> None:
    write_local(tmp_path, "[scan\nprivate_terms = [\n")

    with pytest.raises(ConfigError, match="not valid TOML"):
        read_scan_settings(tmp_path)


def test_an_unknown_scan_key_is_refused(tmp_path: Path) -> None:
    write_local(tmp_path, '[scan]\nprivate_terms = []\nprivate_term = ["typo"]\n')

    with pytest.raises(ConfigError, match="private_term"):
        read_scan_settings(tmp_path)


def test_a_bad_value_in_another_table_is_refused(tmp_path: Path) -> None:
    write_local(tmp_path, '[scan]\nprivate_terms = []\n[git]\nauthor_email = ["typo"]\n')

    with pytest.raises(ConfigError, match="author_email"):
        read_scan_settings(tmp_path)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('scan = "Alicia"', r"\[scan\] must be a table"),
        ('[scan]\nprivate_terms = "Alicia"', r"\[scan\]\.private_terms must be a list of strings"),
        ("[scan]\nprivate_terms = [1]", r"\[scan\]\.private_terms must be a list of strings"),
        ('[scan]\nprivate_terms = ["Alicia", "  "]', r"private_terms entry 2 is blank"),
        ('[scan]\nprivate_terms = []\naccepted = "SPEC.md"', r"accepted must be a list of tables"),
        ("[scan]\nprivate_terms = []\naccepted = [1]", r"accepted entry 1 must be a table"),
        (
            '[scan]\nprivate_terms = []\naccepted = [{ file = "a", match = "b", count = 1, why = "x" }]',
            r"accepted entry 1: unknown key\(s\): why",
        ),
        (
            '[scan]\nprivate_terms = []\naccepted = [{ match = "b" }]',
            r"accepted entry 1 is missing: file, count",
        ),
        (
            '[scan]\nprivate_terms = []\naccepted = [{ file = " ", match = "b", count = 1 }]',
            r"accepted entry 1: file must be a non-empty string",
        ),
        (
            '[scan]\nprivate_terms = []\naccepted = [{ file = "a", match = "", count = 1 }]',
            r"accepted entry 1: match must be a non-empty string",
        ),
        (
            '[scan]\nprivate_terms = []\naccepted = [{ file = "a", match = "b", count = 0 }]',
            r"accepted entry 1: count must be a positive integer",
        ),
        (
            '[scan]\nprivate_terms = []\naccepted = [{ file = "a", match = "b", count = true }]',
            r"accepted entry 1: count must be a positive integer",
        ),
        (
            '[scan]\nprivate_terms = []\naccepted = [{ file = "a", match = "b", count = 1, note = 3 }]',
            r"accepted entry 1: note must be a string",
        ),
        (
            '[scan]\nprivate_terms = []\naccepted = [\n'
            '  { file = "a", match = "Bee", count = 1 },\n'
            '  { file = "a", match = "bee", count = 2 },\n]',
            r"accepted entry 2 repeats an earlier entry's file and match",
        ),
    ],
)
def test_badly_shaped_scan_values_are_refused(tmp_path: Path, body: str, message: str) -> None:
    write_local(tmp_path, body + "\n")

    with pytest.raises(ConfigError, match=message):
        read_scan_settings(tmp_path)


def test_refusals_name_positions_never_values(tmp_path: Path) -> None:
    write_local(
        tmp_path,
        '[scan]\nprivate_terms = ["Alicia"]\n'
        'accepted = [{ file = "SPEC.md", match = "Alicia", count = 0 }]\n',
    )

    with pytest.raises(ConfigError) as exc:
        read_scan_settings(tmp_path)

    assert "entry 1" in str(exc.value)
    assert "alicia" not in str(exc.value).lower()
