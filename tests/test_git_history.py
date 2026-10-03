"""Git history as a source: reading commits, filtering them, and turning them into chunks.

Every repo here is a temp `git init`. Names, emails and secret-shaped strings are
made up, and the secret-shaped ones are assembled at runtime so no literal in this
file looks like a credential.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from second_brain.chunking import Chunk, chunk_document
from second_brain.git_history import (
    Commit,
    GitError,
    collect_repo_history,
    commit_chunks,
    commit_text,
    diff_excerpt,
    filter_commits,
    is_thin,
    is_version_bump,
    read_commits,
    read_diff,
    repo_status,
)

ME = ("Alice Example", "alice@example.com")
TEAMMATE = ("Bob Teammate", "bob@example.org")
DATE = "2026-07-02T10:00:00+05:30"


# --- temp repo helpers -------------------------------------------------------


def git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=.no-hooks", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, **(env or {})},
        check=True,
    )
    return result.stdout.strip()


def init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.name", ME[0])
    git(path, "config", "user.email", ME[1])
    return path


def commit(
    repo: Path,
    message: str,
    files: dict[str, str | bytes] | None = None,
    *,
    author: tuple[str, str] = ME,
    date: str = DATE,
) -> str:
    for rel, content in (files or {}).items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    git(repo, "add", "-A")
    env = {
        "GIT_AUTHOR_NAME": author[0],
        "GIT_AUTHOR_EMAIL": author[1],
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_DATE": date,
    }
    git(repo, "commit", "-q", "--allow-empty", "-m", message, env=env)
    return git(repo, "rev-parse", "HEAD")


def make_commit(
    *,
    subject: str = "feat: add a thing",
    body: str = "",
    files: tuple[str, ...] = ("src/app.py",),
    parents: tuple[str, ...] = ("p" * 40,),
    author: tuple[str, str] = ME,
    date: str = DATE,
    full_hash: str = "b237453" + "0" * 33,
) -> Commit:
    return Commit(
        hash=full_hash,
        parents=parents,
        author_name=author[0],
        author_email=author[1],
        author_date=datetime.fromisoformat(date),
        subject=subject,
        body=body,
        files=files,
    )


# --- read_commits ------------------------------------------------------------


def test_read_commits_parses_every_field(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    root = commit(repo, "Initial commit", {"README.md": "hello\n"})
    second = commit(
        repo,
        "fix: verify ownership\n\nThe endpoint now checks who owns the problem.",
        {"src/api.py": "x = 1\n", "src/model.py": "y = 2\n"},
        date="2026-07-03T09:30:00+05:30",
    )

    commits = read_commits(repo)

    assert [c.hash for c in commits] == [second, root]  # newest first
    newest = commits[0]
    assert newest.subject == "fix: verify ownership"
    assert newest.body == "The endpoint now checks who owns the problem."
    assert newest.author_name == ME[0]
    assert newest.author_email == ME[1]
    assert newest.author_date == datetime.fromisoformat("2026-07-03T09:30:00+05:30")
    assert newest.files == ("src/api.py", "src/model.py")
    assert newest.parents == (root,)
    assert newest.short == second[:7]


def test_root_commit_has_no_parents_and_still_lists_its_files(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    commit(repo, "Initial commit", {"README.md": "hello\n"})

    (root,) = read_commits(repo)

    assert root.parents == ()
    assert root.is_root
    assert root.files == ("README.md",)


def test_merge_commit_has_two_parents(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    commit(repo, "Initial commit", {"README.md": "hello\n"})
    git(repo, "checkout", "-q", "-b", "feature")
    commit(repo, "feat: on a branch", {"feature.py": "f = 1\n"})
    git(repo, "checkout", "-q", "main")
    commit(repo, "feat: on main", {"main.py": "m = 1\n"})
    git(repo, "merge", "-q", "--no-ff", "-m", "Merge branch 'feature'", "feature")

    merge = read_commits(repo)[0]

    assert len(merge.parents) == 2
    assert merge.is_merge


def test_reads_head_only(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    commit(repo, "Initial commit", {"README.md": "hello\n"})
    git(repo, "checkout", "-q", "-b", "unmerged")
    commit(repo, "feat: never merged", {"wip.py": "w = 1\n"})
    git(repo, "checkout", "-q", "main")

    subjects = [c.subject for c in read_commits(repo)]

    assert subjects == ["Initial commit"]


def test_non_ascii_message_survives(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    commit(repo, "docs: café notes · naïve résumé", {"README.md": "hello\n"})

    assert read_commits(repo)[0].subject == "docs: café notes · naïve résumé"


def test_a_directory_that_is_not_a_repo_raises_git_error(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()

    with pytest.raises(GitError):
        read_commits(plain)


def test_a_plain_folder_inside_a_repo_does_not_read_the_enclosing_history(
    tmp_path: Path,
) -> None:
    outer = init_repo(tmp_path / "outer")
    commit(outer, "Initial commit", {"README.md": "hello\n"})
    inner = outer / "not-a-repo"
    inner.mkdir()

    with pytest.raises(GitError):
        read_commits(inner)


# --- filtering -----------------------------------------------------------------


@pytest.mark.parametrize(
    "subject",
    [
        "chore: bump version to 1.1.0",
        "chore: bump version to 1.0.2 for npm release",
        "chore(release): bump version to 2.0.0",
        "1.0.8",
        "v2.3.4",
    ],
)
def test_version_bumps_are_recognised(subject: str) -> None:
    assert is_version_bump(subject)


@pytest.mark.parametrize(
    "subject",
    [
        # Mixed commits that bump a version alongside real work stay in.
        "docs: add install-hook to README and bump version to 1.0.4",
        "fix(cli): read the version from package.json",
        "Fix: Bump SW version to v3 to force a fresh manifest",
        "feat: add 1.0.8 changelog section",
    ],
)
def test_ordinary_commits_are_not_version_bumps(subject: str) -> None:
    assert not is_version_bump(subject)


def test_filter_drops_merges() -> None:
    merge = make_commit(subject="Merge branch 'x'", parents=("a" * 40, "b" * 40))
    normal = make_commit()

    selection = filter_commits([merge, normal], author_emails=[ME[1]], author_names=[])

    assert selection.kept == [normal]
    assert selection.merges == 1


def test_filter_drops_version_bumps() -> None:
    bump = make_commit(subject="chore: bump version to 1.1.0")
    normal = make_commit()

    selection = filter_commits([bump, normal], author_emails=[ME[1]], author_names=[])

    assert selection.kept == [normal]
    assert selection.version_bumps == 1


def test_filter_keeps_an_author_matching_either_list_case_insensitively() -> None:
    by_email = make_commit(author=("Someone Else", "ALICE@example.com"))
    by_name = make_commit(author=("alice example", "alice@other-machine.local"))
    teammate = make_commit(author=TEAMMATE)

    selection = filter_commits(
        [by_email, by_name, teammate], author_emails=[ME[1]], author_names=[ME[0]]
    )

    assert selection.kept == [by_email, by_name]
    assert selection.other_authors == 1


def test_filter_with_empty_author_lists_keeps_nothing() -> None:
    selection = filter_commits([make_commit()], author_emails=[], author_names=[])

    assert selection.kept == []
    assert selection.other_authors == 1


def test_two_author_repo_keeps_only_the_listed_author(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    mine = commit(repo, "feat: my change", {"a.py": "a = 1\n"})
    commit(repo, "feat: their change", {"b.py": "b = 1\n"}, author=TEAMMATE)

    selection = filter_commits(read_commits(repo), author_emails=[ME[1]], author_names=[])

    assert [c.hash for c in selection.kept] == [mine]
    assert selection.other_authors == 1


# --- thin commits ----------------------------------------------------------------


def test_a_commit_without_a_body_is_thin() -> None:
    long_subject = "feat: " + "x" * 300
    assert is_thin(make_commit(subject=long_subject, body=""), thin_message_chars=200)


def test_a_short_message_with_a_body_is_thin() -> None:
    assert is_thin(make_commit(subject="fix: typo", body="Small."), thin_message_chars=200)


def test_a_long_message_with_a_body_is_not_thin() -> None:
    commit_ = make_commit(subject="fix: auth", body="Explains the fix. " * 20)
    assert not is_thin(commit_, thin_message_chars=200)


# --- diff excerpt ----------------------------------------------------------------


def _diff(path: str, *lines: str) -> str:
    return "\n".join(
        [
            f"diff --git a/{path} b/{path}",
            "index 1111111..2222222 100644",
            f"--- a/{path}",
            f"+++ b/{path}",
            "@@ -1 +1 @@",
            *lines,
        ]
    )


def test_excerpt_keeps_changes_and_hunk_headers_but_not_file_headers() -> None:
    excerpt = diff_excerpt(_diff("src/app.py", "-old = 1", "+new = 2"), max_chars=800)

    assert excerpt.splitlines() == ["@@ -1 +1 @@", "-old = 1", "+new = 2"]


def test_excerpt_keeps_a_removed_sql_comment_that_looks_like_a_file_header() -> None:
    # Removing "-- note" yields "--- note" inside the hunk; it is content, not a header.
    diff = _diff("schema.sql", "--- old note", "+-- new note")

    assert diff_excerpt(diff, max_chars=800).splitlines() == [
        "@@ -1 +1 @@",
        "--- old note",
        "+-- new note",
    ]


def test_excerpt_is_capped() -> None:
    diff = _diff("src/app.py", *[f"+line number {i}" for i in range(200)])

    assert len(diff_excerpt(diff, max_chars=800)) <= 800


@pytest.mark.parametrize(
    "path",
    [
        "package-lock.json",
        "web/yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "assets/logo.png",
        "img/photo.JPG",
        "icon.svg",
        ".env",
        ".env.local",
        "config/.env.production",
        "certs/server.pem",
    ],
)
def test_excerpt_skips_files_that_are_noise_or_sensitive(path: str) -> None:
    diff = _diff(path, "+some content here") + "\n" + _diff("src/app.py", "+kept = 1")

    excerpt = diff_excerpt(diff, max_chars=800)

    assert "some content here" not in excerpt
    assert "+kept = 1" in excerpt


def test_excerpt_skips_a_deleted_sensitive_file() -> None:
    diff = "\n".join(
        [
            "diff --git a/.env b/.env",
            "deleted file mode 100644",
            "--- a/.env",
            "+++ /dev/null",
            "@@ -1 +0,0 @@",
            "-SOMETHING=value",
        ]
    )

    assert diff_excerpt(diff, max_chars=800) == ""


def test_excerpt_of_a_binary_change_is_empty() -> None:
    diff = "\n".join(
        [
            "diff --git a/data.bin b/data.bin",
            "new file mode 100644",
            "index 0000000..3333333",
            "Binary files /dev/null and b/data.bin differ",
        ]
    )

    assert diff_excerpt(diff, max_chars=800) == ""


# Assembled at runtime so this file holds no credential-shaped literal.
_GOOGLE_KEY = "AI" + "za" + "Sy" + "Q" * 33
_PRIVATE_KEY_HEADER = "-----BEGIN " + "RSA PRIVATE " + "KEY-----"
_CREDENTIAL_URL = "postgres" + "://admin:" + "hunter2" + "@db.example.com:5432/app"
_BCRYPT = "$2" + "b$10$" + "N" * 53


def _assert_line_dropped(secret_line: str) -> None:
    diff = _diff("src/settings.py", "+before = 1", secret_line, "+after = 2")

    excerpt = diff_excerpt(diff, max_chars=800)

    assert excerpt.splitlines() == ["@@ -1 +1 @@", "+before = 1", "+after = 2"]


def test_secret_filter_drops_a_google_api_key_line() -> None:
    _assert_line_dropped(f'+GEMINI = "{_GOOGLE_KEY}"')


def test_secret_filter_drops_a_private_key_line() -> None:
    _assert_line_dropped("+" + _PRIVATE_KEY_HEADER)


@pytest.mark.parametrize(
    "line",
    [
        "+API_KEY=abc123",
        "+client_secret = 'shh'",
        '+DB_PASSWORD="opensesame"',
        "+password: opensesame",
        "+GITHUB_TOKEN=abc123",
    ],
)
def test_secret_filter_drops_an_assignment_with_a_value(line: str) -> None:
    _assert_line_dropped(line)


def test_secret_filter_keeps_an_assignment_without_a_value() -> None:
    diff = _diff("src/settings.py", '+API_KEY = ""')

    assert '+API_KEY = ""' in diff_excerpt(diff, max_chars=800)


def test_secret_filter_drops_a_credential_url_line() -> None:
    _assert_line_dropped(f'+DATABASE_URL = "{_CREDENTIAL_URL}"')


def test_secret_filter_keeps_a_url_without_credentials() -> None:
    diff = _diff("src/settings.py", '+HOME = "https://example.com/path@v2"')

    assert "https://example.com/path@v2" in diff_excerpt(diff, max_chars=800)


def test_secret_filter_drops_a_bcrypt_hash_line() -> None:
    _assert_line_dropped(f'+seed_hash = "{_BCRYPT}"')


def test_read_diff_returns_the_commits_changes(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    commit(repo, "Initial commit", {"src/app.py": "value = 1\n"})
    second = commit(repo, "fix: value", {"src/app.py": "value = 2\n"})

    diff = read_diff(repo, second)

    assert "-value = 1" in diff
    assert "+value = 2" in diff


# --- commit text -----------------------------------------------------------------


def test_commit_text_has_subject_body_and_files() -> None:
    text = commit_text(
        make_commit(subject="fix: auth", body="Checks ownership.", files=("a.py", "b.py")),
        excerpt="",
    )

    assert text == "fix: auth\n\nChecks ownership.\n\nFiles changed:\na.py\nb.py"


def test_commit_text_without_a_body_skips_the_gap() -> None:
    text = commit_text(make_commit(subject="fix: auth", files=("a.py",)), excerpt="")

    assert text == "fix: auth\n\nFiles changed:\na.py"


def test_commit_text_lists_at_most_30_files() -> None:
    files = tuple(f"src/file{i}.py" for i in range(45))

    text = commit_text(make_commit(files=files), excerpt="")

    assert "src/file29.py" in text
    assert "src/file30.py" not in text
    assert text.endswith("... and 15 more")


def test_commit_text_appends_a_diff_excerpt() -> None:
    text = commit_text(make_commit(), excerpt="@@ -1 +1 @@\n+new = 2")

    assert text.endswith("Diff excerpt:\n@@ -1 +1 @@\n+new = 2")


# --- chunks ------------------------------------------------------------------------


def _chunks(commit_: Commit, text: str | None = None, **kwargs) -> list[Chunk]:
    return commit_chunks(
        commit_,
        commit_text(commit_, excerpt="") if text is None else text,
        project="DSA Tracker",
        repo_path=Path("/projects/DSA Tracker"),
        **kwargs,
    )


def test_commit_chunk_carries_git_metadata() -> None:
    commit_ = make_commit(subject="security: fix submitReview", body="Verifies ownership.")

    (chunk,) = _chunks(commit_)

    assert chunk.source == "git"
    assert chunk.project == "DSA Tracker"
    assert chunk.commit == commit_.hash
    assert chunk.rel_path == "commit b237453"
    assert chunk.heading_path == "security: fix submitReview"
    assert chunk.author == ME[0]
    assert chunk.author_email == ME[1]
    assert chunk.author_date == DATE
    assert chunk.mtime == datetime.fromisoformat(DATE).timestamp()
    assert chunk.path == str(Path("/projects/DSA Tracker"))


def test_commit_chunk_id_is_keyed_on_the_full_hash() -> None:
    commit_ = make_commit(body="Verifies ownership.")

    (chunk,) = _chunks(commit_)

    expected = hashlib.sha256(f"DSA Tracker|git:{commit_.hash}|0".encode()).hexdigest()
    assert chunk.chunk_id == expected


def test_commit_text_never_names_the_repo_but_embed_text_does() -> None:
    commit_ = make_commit(subject="security: fix submitReview", body="Verifies ownership.")

    (chunk,) = _chunks(commit_)

    assert "DSA Tracker" not in chunk.text
    assert chunk.embed_text.startswith(
        "DSA Tracker · commit b237453 · 2026-07-02\nsecurity: fix submitReview"
    )
    assert chunk.content_hash == hashlib.sha256(chunk.embed_text.encode()).hexdigest()


def test_every_piece_of_a_long_commit_names_repo_and_subject() -> None:
    body = "\n\n".join(f"Paragraph {i}. " + "detail " * 40 for i in range(12))
    commit_ = make_commit(subject="refactor: split the scheduler", body=body)

    chunks = _chunks(commit_)

    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert len(chunk.text) <= 1500
        assert chunk.embed_text.startswith("DSA Tracker · commit b237453 · 2026-07-02\n")
        assert "refactor: split the scheduler" in chunk.embed_text.split("\n\n")[0]


def test_a_trivial_root_commit_falls_below_the_floor() -> None:
    root = make_commit(subject="Initial commit", files=("README.md",), parents=())

    assert _chunks(root) == []


def test_document_chunks_are_marked_as_docs(doc_factory) -> None:
    (chunk,) = chunk_document(doc_factory(), "# Title\n\nA body that is long enough to clear the size floor.")

    assert chunk.source == "doc"
    assert chunk.commit == ""


# --- one repo, end to end ----------------------------------------------------------


def test_collect_repo_history_end_to_end(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    # Trivial root: no excerpt (root), and too short to survive the floor.
    commit(repo, "Initial commit", {"README.md": "hello\n"})
    # Thin, non-root: gets a diff excerpt.
    thin = commit(repo, "fix: off by one", {"src/loop.py": "for i in range(10):\n    pass\n"})
    # Not thin: a long body and no excerpt.
    detailed = commit(
        repo,
        "feat: scheduler\n\n" + "Describes the scheduler design in detail. " * 8,
        {"src/scheduler.py": "SCHEDULE = []\n"},
    )
    commit(repo, "chore: bump version to 1.0.1", {"version.txt": "1.0.1\n"})
    commit(repo, "feat: teammate work", {"src/other.py": "o = 1\n"}, author=TEAMMATE)

    history = collect_repo_history(
        repo,
        project="Repo",
        author_emails=[ME[1]],
        author_names=[],
        thin_message_chars=200,
        diff_excerpt_chars=800,
    )

    assert history.commits == 5
    assert history.kept == 3
    assert history.version_bumps == 1
    assert history.other_authors == 1
    assert history.merges == 0
    assert history.with_excerpt == 1

    by_commit = {c.commit: c for c in history.chunks}
    assert set(by_commit) == {thin, detailed}
    assert "Diff excerpt:" in by_commit[thin].text
    assert "+for i in range(10):" in by_commit[thin].text
    assert "Diff excerpt:" not in by_commit[detailed].text


# --- repo status (for the dry run) -------------------------------------------------


def _clone_with_origin(tmp_path: Path) -> Path:
    upstream = init_repo(tmp_path / "upstream")
    commit(upstream, "Initial commit", {"README.md": "hello\n"})
    bare = tmp_path / "origin.git"
    git(tmp_path, "clone", "-q", "--bare", str(upstream), str(bare))
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(bare), str(clone))
    git(clone, "config", "user.name", ME[0])
    git(clone, "config", "user.email", ME[1])
    return clone


def test_status_of_a_clean_clone_on_its_default_branch_has_no_flags(tmp_path: Path) -> None:
    clone = _clone_with_origin(tmp_path)

    status = repo_status(clone)

    assert status.branch == "main"
    assert status.default_branch == "main"
    assert status.unpushed == 0
    assert status.flags == []


def test_status_flags_a_repo_off_its_default_branch(tmp_path: Path) -> None:
    clone = _clone_with_origin(tmp_path)
    git(clone, "checkout", "-q", "-b", "experiment")

    status = repo_status(clone)

    assert status.branch == "experiment"
    assert any("experiment" in flag and "main" in flag for flag in status.flags)


def test_status_counts_unpushed_commits(tmp_path: Path) -> None:
    clone = _clone_with_origin(tmp_path)
    commit(clone, "feat: local only", {"local.py": "l = 1\n"})
    commit(clone, "feat: also local", {"local2.py": "l = 2\n"})

    status = repo_status(clone)

    assert status.unpushed == 2
    assert any("2 unpushed" in flag for flag in status.flags)


def test_status_flags_a_repo_with_no_remote(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    commit(repo, "Initial commit", {"README.md": "hello\n"})

    status = repo_status(repo)

    assert not status.has_remote
    assert status.flags == ["no remote"]


def test_status_flags_a_detached_head(tmp_path: Path) -> None:
    clone = _clone_with_origin(tmp_path)
    git(clone, "checkout", "-q", "--detach")

    status = repo_status(clone)

    assert status.branch is None
    assert "detached HEAD" in status.flags
