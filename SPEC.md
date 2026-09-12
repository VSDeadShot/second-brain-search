# Second-Brain Search — v1 Spec

## Problem
Vedansh has shipped 10+ projects (Watch Next, DSA Tracker, Interview Prep Flashcards, Now Brief, OmniTask, commit-gen, Bixby PC Navigator, the mandala art site, AI Macro Tracker, plus SIH's Driftless work) each with their own READMEs, CLAUDE_SUMMARY.md changelogs, and spec docs. There's no way to ask "how did I solve X in project Y" without manually digging through the right repo. This project builds a semantic search + RAG layer over all of that self-authored documentation.

## Goals (v1)
- Ingest documentation (README.md, CLAUDE_SUMMARY.md, PROJECT_STATUS.md, and any docs/specs/*.md files) from a configurable list of local repo paths
- Chunk and embed that content into a local vector store
- Query via CLI: ask a natural-language question, get a RAG-generated answer citing which project/file it came from
- Re-indexing is a manual, explicit command (no file-watching in v1)

## Non-goals (v1)
- No web UI (CLI first; a Now Brief dashboard card is a plausible v2 extension, not v1 scope)
- No code-level indexing (source files), only documentation/markdown — keeps scope tight and avoids license/noise issues from third-party code in dependencies
- No multi-user / auth — this is a single-user local tool

## Proposed stack (open for discussion, not locked)
- **Language:** Python (deliberately — addresses the GitHub language-mix gap)
- **Embeddings:** Gemini embedding API (reuses the same API-key pattern already used in Macro Tracker / Flashcards) — local sentence-transformers is the fallback if API cost/rate limits become a problem
- **Vector store:** ChromaDB (local, embedded, no server to run — fits a single-user CLI tool)
- **RAG answer generation:** Gemini, same key as embeddings
- **CLI framework:** Click or argparse

## Decisions (resolved Sep 9)
1. **Repo scope** — scan a single parent "Projects" folder automatically (not a manually-maintained config list, so new projects are covered with zero upkeep). Skip `node_modules`, `.git`, `venv`, `build`, `dist` and similar dependency/build directories. Only pull the target file patterns (README.md, CLAUDE_SUMMARY.md, PROJECT_STATUS.md, docs/specs/*.md) — not every markdown file encountered.
2. **Embedding choice** — Gemini API.
3. **Chunking strategy** — split large docs into smaller chunks rather than one-chunk-per-file, for better retrieval precision.
4. **Answer format** — every answer always cites which project/file it was pulled from.
5. **Re-index trigger** — full rebuild on each `index`/`reindex` command for v1 (no incremental-by-mtime logic yet — at this project's scale a full rebuild is fast, and incremental updates are a natural v2 addition once actually needed, not before).

## v1 feature slices (build one at a time, confirm working before moving to the next)
1. Config + repo discovery (read a list of paths, find target doc files)
2. Chunking + embedding pipeline (embed and store in Chroma)
3. `index` CLI command (run the pipeline end-to-end, report what got indexed)
4. Retrieval (given a query, return top-k relevant chunks)
5. `ask` CLI command (retrieval + Gemini RAG answer with citations)
6. Re-index command + basic incremental-update handling
