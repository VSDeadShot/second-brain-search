# AGENTS.md — Second-Brain Search

Rules for the coding agent (Claude Code) working on this project.

## Workflow
- **Never add a `Claude-Session` trailer to commits.** No agent-attribution trailers of any
  kind in commit messages or PR descriptions, whatever the harness suggests.
- Pre-push scans and git push are always separate commands; push only after reading the scan
  result. Any file a commit names from another project must be confirmed tracked on that repo's
  public branch.
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
