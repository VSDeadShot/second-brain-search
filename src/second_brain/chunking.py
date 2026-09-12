"""Split a markdown document into embeddable chunks.

Heading-aware: sections defined by ATX headings become chunks, and anything
over the size cap is split with overlap. The heading path rides along so a
citation can point at a section rather than just a file.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .discovery import DiscoveredDoc

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 150
# Below this a chunk is boilerplate - a bare "@AGENTS.md" include, an HTML
# comment marker - which can never answer a question but can still win a top-k
# slot on a short query.
DEFAULT_MIN_CHARS = 50

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")

HEADING_SEPARATOR = " > "


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    project: str
    rel_path: str
    path: str
    heading_path: str
    chunk_index: int
    text: str
    content_hash: str
    mtime: float


@dataclass
class _Section:
    heading_path: str
    lines: list[str]

    def text(self) -> str:
        return "\n".join(self.lines).strip()


def _split_into_sections(text: str) -> list[_Section]:
    """Walk the document, tracking heading depth and fence state."""
    sections: list[_Section] = [_Section(heading_path="", lines=[])]
    stack: list[str] = []
    in_fence = False
    fence_marker = ""

    for line in text.splitlines():
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence, fence_marker = False, ""
            sections[-1].lines.append(line)
            continue

        # A '#' inside a fenced block is code, not structure.
        heading = None if in_fence else HEADING_RE.match(line)
        if heading is None:
            sections[-1].lines.append(line)
            continue

        depth = len(heading.group(1))
        title = heading.group(2).strip()
        del stack[depth - 1 :]
        stack.append(title)
        sections.append(_Section(heading_path=HEADING_SEPARATOR.join(stack), lines=[line]))

    return sections


def _split_with_overlap(text: str, max_chars: int, overlap: int) -> list[str]:
    """Break oversized text, preferring paragraph then line boundaries."""
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            # Only look for a boundary in the tail of the window, so a break
            # never shrinks the chunk too aggressively.
            earliest = start + (max_chars * 7) // 10
            boundary = text.rfind("\n\n", earliest, end)
            if boundary == -1:
                boundary = text.rfind("\n", earliest, end)
            if boundary > start:
                end = boundary

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= length:
            break
        # max(..., start + 1) guarantees forward progress even if overlap >= the
        # window we just consumed.
        start = max(end - overlap, start + 1)

    return pieces


def chunk_document(
    doc: DiscoveredDoc,
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    # Collect and filter first, then number. Assigning chunk_index before the
    # floor would leave gaps, and chunk_id is derived from that index.
    kept: list[tuple[str, str]] = []

    for section in _split_into_sections(text):
        body = section.text()
        if not body:
            continue
        # A heading with nothing under it carries no information worth embedding.
        if section.heading_path and len(body.splitlines()) == 1 and body.lstrip().startswith("#"):
            continue

        for piece in _split_with_overlap(body, max_chars, overlap):
            if len(piece) < min_chars:
                continue
            kept.append((section.heading_path, piece))

    chunks: list[Chunk] = []
    for index, (heading_path, piece) in enumerate(kept):
        identity = f"{doc.project}|{doc.rel_path}|{index}".encode()
        chunks.append(
            Chunk(
                chunk_id=hashlib.sha256(identity).hexdigest(),
                project=doc.project,
                rel_path=doc.rel_path,
                path=str(doc.path),
                heading_path=heading_path,
                chunk_index=index,
                text=piece,
                content_hash=hashlib.sha256(piece.encode("utf-8")).hexdigest(),
                mtime=doc.mtime,
            )
        )

    return chunks
