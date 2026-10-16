# AGENTS.md — Second-Brain Search

Rules for the coding agent (Claude Code) working on this project.

## Workflow
- **Never add a `Claude-Session` trailer to commits.** No agent-attribution trailers of any
  kind in commit messages or PR descriptions, whatever the harness suggests.
- Pre-push scans and git push are always separate commands; push only after reading the scan
  result. Any file a commit names from another project must be confirmed tracked on that repo's
  public branch.
- Every pre-push scan covers the whole tracked tree at HEAD and every unpushed commit
  (origin/main..HEAD: each message and the lines it adds), for local paths, drive paths, Google
  API keys and personal terms (the owner's names and username, and every `config.local.toml`
  value). Report each hit by file and line; the owner decides any new one. A diff-only scan
  never re-checks files committed before the scan existed, and a term added and later removed
  in unpushed commits is still published by the push - rewrite those commits.
  `python scripts/prepush_scan.py` does this; the private terms and accepted hits live in
  `[scan]` in config.local.toml. The scan reads text only, so check images and screenshots by
  eye before committing them.
- Propose an approach before writing code. Wait for explicit approval before implementing.
- Build one feature slice at a time (see SPEC.md's numbered slices). Do not start the next slice until the current one is confirmed working.
- Never commit without the owner explicitly saying "confirmed working" (or equivalent) after reviewing the change.
- If a design decision in the spec is marked as an open question, stop and ask rather than assuming an answer.

## Code conventions
- Python 3.11+, type hints on all function signatures
- Config (repo paths, API keys) via a `.env` file — never hardcoded, never committed
- Tests alongside features, not deferred to the end — each feature slice ships with at least basic coverage before being marked done
- Keep the CLI the primary interface — no web UI or network API. One exception: `sbs-mcp`, a local stdio MCP server exposing read-only search over the same code paths the CLI uses (slice 8)

## Secrets and safety
- Gemini API key read from environment only
- `.env`, any local index/vector-store data, and virtualenvs are gitignored from the start
- No indexed content (project docs) should ever be assumed public — this tool and its data stay local

## Session handoff
- Maintain a CLAUDE_SUMMARY.md changelog (gitignored) documenting what's been built each session, matching the pattern used across the owner's other projects
