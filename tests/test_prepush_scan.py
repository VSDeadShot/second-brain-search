"""scripts/prepush_scan.py: the whole-tracked-tree leak check run before every push."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepush_scan.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepush_scan", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered first: dataclasses look the module up while it executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepush_scan = _load_script()

TERMS = '[scan]\nprivate_terms = ["Alicia", "licia"]\n'

# The scan reads this file too, so fixture paths and emails are assembled at runtime:
# the source itself carries no drive path, home path or real-looking email.
C = "C" + ":"


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def make_repo(tmp_path: Path, files: dict[str, str | bytes], local: str = TERMS) -> Path:
    """A committed repo holding `files`, with an untracked config.local.toml."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "core.autocrlf", "false")
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    git(repo, "add", "--", *files)
    git(repo, "commit", "-q", "-m", "init")
    (repo / "config.local.toml").write_text(local, encoding="utf-8")
    return repo


def run(repo: Path, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    """Exit code and everything printed - refusals go to stderr."""
    code = prepush_scan.main(["--repo", str(repo), *args])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_a_clean_tree_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = make_repo(tmp_path, {"README.md": "# Demo\n\nNothing private here.\n"})

    code, out = run(repo, capsys)

    assert code == 0
    assert "new 0" in out


def test_a_term_hit_is_reported_by_file_and_line_never_by_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"docs/notes.md": "line one\nwritten by ALICIA\n"})

    code, out = run(repo, capsys)

    assert code == 1
    assert "NEW" in out and "docs/notes.md:2" in out and "term #0" in out
    assert "alicia" not in out.lower()


def test_a_term_inside_a_longer_term_is_reported_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"a.md": "Alicia\n"})

    _, out = run(repo, capsys)

    assert out.count("a.md:1") == 1
    assert "term #1" not in out


def test_private_values_from_the_other_tables_are_scanned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    local = TERMS + '[mcp]\nexclude_projects = ["Hidden Thing"]\n'
    repo = make_repo(tmp_path, {"a.md": "see hidden thing\n"}, local=local)

    code, out = run(repo, capsys)

    assert code == 1
    assert "a.md:1" in out and "term #2" in out
    assert "hidden thing" not in out.lower()


def test_head_is_scanned_not_uncommitted_edits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"})
    (repo / "a.md").write_text("clean\nAlicia\n", encoding="utf-8")

    assert run(repo, capsys)[0] == 0
    code, out = run(repo, capsys, "--worktree")
    assert code == 1 and "a.md:2" in out


def test_rev_selects_the_commit_scanned(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = make_repo(tmp_path, {"a.md": "Alicia\n"})
    (repo / "a.md").write_text("clean\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", "clean")

    assert run(repo, capsys)[0] == 0
    assert run(repo, capsys, "--rev", "HEAD~1")[0] == 1


# --- accepted hits -----------------------------------------------------------------

ACCEPT_ONE = TERMS + 'accepted = [{ file = "SPEC.md", match = "Alicia", count = 1, note = "kept" }]\n'


def test_an_accepted_hit_is_shown_and_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = make_repo(tmp_path, {"SPEC.md": "By Alicia.\n"}, local=ACCEPT_ONE)

    code, out = run(repo, capsys)

    assert code == 0
    assert "ACCEPTED" in out and "SPEC.md:1" in out and "kept" in out
    assert "alicia" not in out.lower()


def test_ignore_accepted_reports_the_accepted_hit_as_new(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"SPEC.md": "By Alicia.\n"}, local=ACCEPT_ONE)

    code, out = run(repo, capsys, "--ignore-accepted")

    assert code == 1
    assert "NEW" in out and "SPEC.md:1" in out


def test_more_lines_than_accepted_are_all_new(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"SPEC.md": "By Alicia.\nAlicia again.\n"}, local=ACCEPT_ONE)

    code, out = run(repo, capsys)

    assert code == 1
    assert "SPEC.md:1" in out and "SPEC.md:2" in out
    assert "ACCEPTED" not in out


def test_a_stale_accepted_entry_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = make_repo(tmp_path, {"SPEC.md": "No name any more.\n"}, local=ACCEPT_ONE)

    code, out = run(repo, capsys)

    assert code == 1
    assert "STALE" in out and "accepted entry 1" in out
    assert "alicia" not in out.lower()


# --- generic patterns --------------------------------------------------------------


def test_placeholder_emails_pass_and_real_ones_do_not(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    placeholder = make_repo(
        tmp_path / "p", {"a.md": "you@example.com\nx@y.invalid\nz@box.local\nq@example.org\n"}
    )
    real = make_repo(tmp_path / "r", {"a.md": "someone" + "@corp-mail.com\n"})

    code, out = run(placeholder, capsys)
    assert code == 0 and "placeholder emails 4" in out
    code, out = run(real, capsys)
    assert code == 1 and "a.md:1" in out and "email" in out


@pytest.mark.parametrize(
    "text",
    [f"{C}/work/repo/README.md", "D" + ":\\data\\x.txt", "/" + "home/someone/x", "/" + "Users/someone/x"],
)
def test_drive_and_home_paths_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], text: str
) -> None:
    repo = make_repo(tmp_path, {"a.md": f"path: {text}\n"})

    code, out = run(repo, capsys)

    assert code == 1 and "a.md:1" in out


def test_an_accepted_drive_path_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = f"{C}/x/README.md"
    local = TERMS + f'accepted = [{{ file = "t.py", match = "{path}", count = 2 }}]\n'
    repo = make_repo(tmp_path, {"t.py": f'a = "{path}"\nb = "{path}"\n'}, local=local)

    code, out = run(repo, capsys)

    assert code == 0 and out.count("ACCEPTED") == 2


def test_a_term_inside_a_path_is_masked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = make_repo(tmp_path, {"a.md": C + "/" + "Users/alicia/notes.md\n"})

    code, out = run(repo, capsys)

    assert code == 1 and "[term #0]" in out
    assert "alicia" not in out.lower()


def test_a_term_in_a_tracked_file_name_is_reported_masked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"docs/alicia-notes.md": "clean\n"})

    code, out = run(repo, capsys)

    assert code == 1 and "file name" in out and "[term #0]" in out
    assert "alicia" not in out.lower()


def test_binary_files_are_listed_as_not_scanned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"shot.png": b"\x89PNG\r\n\x00\x00Alicia\x00"})

    code, out = run(repo, capsys)

    assert code == 0
    assert "not scanned" in out and "shot.png" in out


# --- refusals ----------------------------------------------------------------------


def test_a_missing_scan_config_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"}, local='[git]\nauthor_emails = ["a@example.com"]\n')

    code, out = run(repo, capsys)

    assert code == 2 and "[scan]" in out


def test_an_unknown_rev_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"})

    assert run(repo, capsys, "--rev", "no-such-rev")[0] == 2
