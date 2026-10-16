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


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(repo: Path, files: dict[str, str | bytes]) -> None:
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")


def make_repo(tmp_path: Path, files: dict[str, str | bytes], local: str = TERMS) -> Path:
    """A repo holding `files`, committed and pushed to a local bare origin (nothing
    unpushed), with an untracked config.local.toml."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "remote", "add", "origin", str(origin))
    _write(repo, files)
    git(repo, "add", "--", *files)
    git(repo, "commit", "-q", "-m", "init")
    push(repo)
    (repo / "config.local.toml").write_text(local, encoding="utf-8")
    return repo


def commit(repo: Path, files: dict[str, str | bytes], message: str) -> str:
    """Commit `files` without pushing; returns the short hash the scan prints."""
    _write(repo, files)
    if files:
        git(repo, "add", "--", *files)
    git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return git(repo, "rev-parse", "HEAD")[:7]


def push(repo: Path) -> None:
    git(repo, "push", "-q", "origin", "main")


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


# --- unpushed commits (origin/main..HEAD) -------------------------------------------


def test_nothing_unpushed_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"})

    code, out = run(repo, capsys)

    assert code == 0
    assert "commits: 0 unpushed" in out and "nothing to scan" in out


def test_a_term_only_in_an_unpushed_message_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"})
    short = commit(repo, {"b.md": "also clean\n"}, "tidy\n\nAsked by Alicia.")

    code, out = run(repo, capsys)

    assert code == 1
    assert f"NEW       {short} message:3  term #0" in out
    assert "alicia" not in out.lower()


def test_a_term_added_then_removed_in_unpushed_commits_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"})
    added = commit(repo, {"a.md": "clean\nby Alicia\n"}, "add a line")
    commit(repo, {"a.md": "clean\n"}, "drop it again")

    code, out = run(repo, capsys)

    assert code == 1  # HEAD's tree is clean; the history would still publish it
    assert f"{added} +a.md:2  term #0" in out
    assert "alicia" not in out.lower()


def test_a_term_in_a_file_name_renamed_away_still_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"})
    added = commit(repo, {"docs/alicia-notes.md": "fine\n"}, "add notes")
    git(repo, "mv", "docs/alicia-notes.md", "docs/notes.md")
    git(repo, "commit", "-q", "-m", "rename")

    code, out = run(repo, capsys)

    assert code == 1
    assert f"{added} +docs/[term #0]-notes.md  term #0 in file name" in out
    assert "alicia" not in out.lower()


def test_pushed_commits_are_not_rescanned(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"})
    commit(repo, {"a.md": "by Alicia\n"}, "Alicia wrote this")
    commit(repo, {"a.md": "clean\n"}, "tidy")
    push(repo)  # already public: nothing a scan now can unpublish
    commit(repo, {"b.md": "fine\n"}, "more")

    code, out = run(repo, capsys)

    assert code == 0
    assert "commits: 1 unpushed" in out


def test_positive_control_a_planted_message_term_fails_until_reworded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"})
    commit(repo, {}, "planted: alicia")

    assert run(repo, capsys)[0] == 1
    git(repo, "commit", "-q", "--amend", "--allow-empty", "-m", "reworded")
    code, out = run(repo, capsys)
    assert code == 0 and "commits: 1 unpushed" in out


def test_a_missing_upstream_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"})
    git(repo, "update-ref", "-d", "refs/remotes/origin/main")

    code, out = run(repo, capsys)

    assert code == 2 and "origin/main" in out


def test_upstream_option_sets_the_range_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"})
    commit(repo, {}, "Alicia")
    commit(repo, {}, "fine")

    assert run(repo, capsys)[0] == 1
    code, out = run(repo, capsys, "--upstream", "HEAD~1")
    assert code == 0 and "commits: 1 unpushed" in out


def test_a_diverged_branch_warns_and_scans_its_local_commits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"})
    local = commit(repo, {}, "from Alicia")
    remote_side = git(repo, "commit-tree", "HEAD~1^{tree}", "-p", "origin/main", "-m", "elsewhere")
    git(repo, "update-ref", "refs/remotes/origin/main", remote_side)

    code, out = run(repo, capsys)

    assert code == 1
    assert "diverged: 1 ahead, 1 behind" in out
    assert f"{local} message:1  term #0" in out


def test_an_accepted_file_hit_in_a_commit_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"SPEC.md": "By Alicia.\n"}, local=ACCEPT_ONE)
    short = commit(repo, {"SPEC.md": "Written by Alicia.\n"}, "reword the byline")

    code, out = run(repo, capsys)

    assert code == 0
    assert f"ACCEPTED  {short} +SPEC.md:1  term #0  (kept)" in out


def test_a_second_accepted_term_added_then_removed_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"SPEC.md": "By Alicia.\n"}, local=ACCEPT_ONE)
    added = commit(repo, {"SPEC.md": "By Alicia.\nAlicia again.\n"}, "add")
    commit(repo, {"SPEC.md": "By Alicia.\n"}, "remove")

    code, out = run(repo, capsys)

    assert code == 1
    assert f"NEW       {added} +SPEC.md:2  term #0" in out


def test_accepted_entries_never_cover_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"SPEC.md": "By Alicia.\n"}, local=ACCEPT_ONE)
    commit(repo, {}, "SPEC.md: Alicia")

    assert run(repo, capsys)[0] == 1


def test_ignore_accepted_applies_to_commit_hits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"SPEC.md": "By Alicia.\n"}, local=ACCEPT_ONE)
    short = commit(repo, {"SPEC.md": "Written by Alicia.\n"}, "reword the byline")

    code, out = run(repo, capsys, "--ignore-accepted")

    assert code == 1 and f"NEW       {short} +SPEC.md:1  term #0" in out


def test_a_binary_file_in_a_commit_is_listed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"})
    short = commit(repo, {"shot.png": b"\x89PNG\r\n\x00\x00Alicia\x00"}, "add a screenshot")

    code, out = run(repo, capsys)

    assert code == 0
    assert f"BINARY    {short} +shot.png  not scanned - check it by eye" in out


def test_a_merge_commit_message_is_scanned(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"})
    git(repo, "checkout", "-q", "-b", "side")
    commit(repo, {"b.md": "side\n"}, "side work")
    git(repo, "checkout", "-q", "main")
    commit(repo, {"c.md": "main\n"}, "main work")
    git(repo, "merge", "-q", "--no-ff", "side", "-m", "merge for Alicia")
    merge = git(repo, "rev-parse", "HEAD")[:7]

    code, out = run(repo, capsys)

    assert code == 1
    assert f"{merge} message:1  term #0" in out
    assert f"MERGE     {merge}  diff not scanned (message scanned)" in out


# --- Google API keys ---------------------------------------------------------------

# Assembled at runtime, so this file's own source never matches the pattern.
FAKE_KEY = "AIza" + "B" * 35


def test_a_fake_google_key_in_an_unpushed_message_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"})
    short = commit(repo, {}, f"set the key to {FAKE_KEY}")

    code, out = run(repo, capsys)

    assert code == 1
    assert f"NEW       {short} message:1  google api key" in out
    assert FAKE_KEY not in out


def test_a_fake_google_key_in_the_tree_fails_and_is_never_accepted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    local = TERMS + f'accepted = [{{ file = "a.md", match = "{FAKE_KEY}", count = 1 }}]\n'
    repo = make_repo(tmp_path, {"a.md": f"key = {FAKE_KEY}\n"}, local=local)

    code, out = run(repo, capsys)

    assert code == 1
    assert "NEW       a.md:1  google api key" in out
    assert FAKE_KEY not in out
