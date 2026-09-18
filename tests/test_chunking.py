"""Chunking: heading-aware, size-capped, with overlap."""

import hashlib
from pathlib import Path

import pytest

from second_brain.chunking import chunk_document


def test_short_document_is_one_chunk(doc_factory) -> None:
    chunks = chunk_document(doc_factory(), "# Title\n\nA short body.", min_chars=0)

    assert len(chunks) == 1
    assert "A short body." in chunks[0].text


def test_each_heading_starts_a_new_chunk(doc_factory) -> None:
    text = "# One\n\nfirst body\n\n# Two\n\nsecond body"

    chunks = chunk_document(doc_factory(), text, min_chars=0)

    assert len(chunks) == 2
    assert "first body" in chunks[0].text
    assert "second body" in chunks[1].text


def test_heading_path_records_nesting(doc_factory) -> None:
    text = "# Architecture\n\nintro\n\n## Caching\n\ndetail here"

    chunks = chunk_document(doc_factory(), text, min_chars=0)

    assert chunks[0].heading_path == "Architecture"
    assert chunks[1].heading_path == "Architecture > Caching"


def test_sibling_heading_replaces_rather_than_nests(doc_factory) -> None:
    text = "# A\n\nx\n\n## B\n\ny\n\n## C\n\nz"

    paths = [c.heading_path for c in chunk_document(doc_factory(), text, min_chars=0)]

    assert paths == ["A", "A > B", "A > C"]


def test_content_before_the_first_heading_is_kept(doc_factory) -> None:
    text = "preamble text here\n\n# Later\n\nbody"

    chunks = chunk_document(doc_factory(), text, min_chars=0)

    assert "preamble text here" in chunks[0].text
    assert chunks[0].heading_path == ""


def test_hashes_inside_code_fences_are_not_headings(doc_factory) -> None:
    """A '# comment' inside a fenced block must not split the document."""
    text = "# Real\n\n```python\n# not a heading\nx = 1\n# also not\n```\n\ntail"

    chunks = chunk_document(doc_factory(), text, min_chars=0)

    assert len(chunks) == 1
    assert "x = 1" in chunks[0].text


def test_oversized_section_is_split_with_overlap(doc_factory) -> None:
    body = "\n\n".join(f"paragraph number {i} with some filler text" for i in range(80))
    text = f"# Big\n\n{body}"

    chunks = chunk_document(doc_factory(), text, max_chars=500, overlap=100)

    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)


def test_split_chunks_share_the_heading_path(doc_factory) -> None:
    body = "\n\n".join(f"line {i} padding padding padding" for i in range(60))
    text = f"# Deep\n\n## Section\n\n{body}"

    chunks = chunk_document(doc_factory(), text, max_chars=400, overlap=80)
    split = [c for c in chunks if c.heading_path == "Deep > Section"]

    assert len(split) > 1


def test_chunk_index_is_sequential(doc_factory) -> None:
    text = "# A\n\nx\n\n# B\n\ny\n\n# C\n\nz"

    chunks = chunk_document(doc_factory(), text, min_chars=0)

    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_chunk_ids_are_deterministic(doc_factory) -> None:
    doc = doc_factory()
    text = "# A\n\nx\n\n# B\n\ny"

    first = [c.chunk_id for c in chunk_document(doc, text, min_chars=0)]
    second = [c.chunk_id for c in chunk_document(doc, text, min_chars=0)]

    assert first == second


def test_chunk_ids_are_unique_within_a_document(doc_factory) -> None:
    text = "# A\n\nx\n\n# B\n\ny\n\n# C\n\nz"

    ids = [c.chunk_id for c in chunk_document(doc_factory(), text, min_chars=0)]

    assert len(ids) == len(set(ids))


def test_chunk_ids_differ_across_documents(doc_factory) -> None:
    text = "# Same\n\nidentical body"

    a = chunk_document(doc_factory("README.md", "Alpha"), text, min_chars=0)[0]
    b = chunk_document(doc_factory("README.md", "Beta"), text, min_chars=0)[0]

    assert a.chunk_id != b.chunk_id


def test_citation_fields_are_carried(doc_factory) -> None:
    doc = doc_factory("docs/guide.md", "Alpha")

    chunk = chunk_document(doc, "# T\n\nbody", min_chars=0)[0]

    assert chunk.project == "Alpha"
    assert chunk.rel_path == "docs/guide.md"
    assert chunk.mtime == doc.mtime
    assert chunk.content_hash


def test_empty_document_yields_no_chunks(doc_factory) -> None:
    assert chunk_document(doc_factory(), "   \n\n  \n", min_chars=0) == []


def test_headings_with_no_body_are_dropped(doc_factory) -> None:
    """A bare heading carries no information worth embedding."""
    text = "# Empty\n\n# Real\n\nactual content"

    chunks = chunk_document(doc_factory(), text, min_chars=0)

    assert len(chunks) == 1
    assert chunks[0].heading_path == "Real"


def test_near_empty_chunks_are_dropped(doc_factory) -> None:
    """Real corpus noise: a bare '@AGENTS.md' include can never answer a question."""
    body = "substantial body content that easily clears the floor. " * 3
    text = f"@AGENTS.md\n\n# Real Section\n\n{body}"

    chunks = chunk_document(doc_factory(), text)

    assert all(len(c.text) >= 50 for c in chunks)
    assert not any(c.text.strip() == "@AGENTS.md" for c in chunks)


def test_min_chars_is_configurable(doc_factory) -> None:
    text = "short preamble\n\n# Heading\n\n" + ("padding text here. " * 10)

    unfiltered = chunk_document(doc_factory(), text, min_chars=0)
    filtered = chunk_document(doc_factory(), text, min_chars=50)

    assert len(unfiltered) == len(filtered) + 1


def test_chunk_index_stays_contiguous_after_filtering(doc_factory) -> None:
    """Ids derive from chunk_index, so a dropped chunk must not leave a gap."""
    big = "long enough body text to survive the floor comfortably. " * 3
    text = f"tiny\n\n# A\n\n{big}\n\n# B\n\n{big}"

    chunks = chunk_document(doc_factory(), text)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len(chunks) == 2


# --- heading context for embedding ----------------------------------------------
#
# Only the first piece of a split section contains its heading line; on the real
# corpus 317 of 812 chunks (39%) carried no heading text at all, so a chunk deep
# in "Caching" might never mention caching. Those chunks are embedded with their
# heading path prepended. The displayed text is unchanged.


def long_section(heading_lines: str) -> str:
    paragraph = "A paragraph about the behaviour that runs on for a while. " * 3
    return heading_lines + "\n\n" + "\n\n".join([paragraph] * 12)


def test_headless_pieces_embed_with_their_heading_path(doc_factory) -> None:
    text = long_section("# Architecture\n\n## Caching")

    chunks = chunk_document(doc_factory(), text)
    headless = [c for c in chunks if not c.text.lstrip().startswith("#")]

    assert headless, "fixture must produce a split piece without its heading"
    for chunk in headless:
        assert chunk.embed_text == f"Architecture > Caching\n\n{chunk.text}"


def test_pieces_that_start_with_their_heading_embed_as_is(doc_factory) -> None:
    chunks = chunk_document(doc_factory(), long_section("# Architecture\n\n## Caching"))
    headed = [c for c in chunks if c.text.lstrip().startswith("#")]

    assert headed
    assert all(c.embed_text == c.text for c in headed)


def test_display_text_is_never_prefixed(doc_factory) -> None:
    chunks = chunk_document(doc_factory(), long_section("# Architecture\n\n## Caching"))

    assert not any(c.text.startswith("Architecture > Caching") for c in chunks)


def test_text_before_any_heading_embeds_as_is(doc_factory) -> None:
    text = "An opening paragraph that sits above every heading in the file.\n\n# Later\n\nbody"

    preamble = chunk_document(doc_factory(), text, min_chars=0)[0]

    assert preamble.heading_path == ""
    assert preamble.embed_text == preamble.text


def test_content_hash_covers_the_embedded_text(doc_factory) -> None:
    """The hash is the embedding-cache key: it must change when what is embedded
    changes, including a heading rename over an identical body."""
    before = chunk_document(doc_factory(), long_section("# Architecture\n\n## Caching"))
    after = chunk_document(doc_factory(), long_section("# Architecture\n\n## Memoisation"))

    old = next(c for c in before if not c.text.lstrip().startswith("#"))
    new = next(c for c in after if c.chunk_index == old.chunk_index)

    assert old.text == new.text
    assert old.content_hash != new.content_hash
    assert old.content_hash == hashlib.sha256(old.embed_text.encode("utf-8")).hexdigest()
