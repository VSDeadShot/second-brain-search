"""Embeddings saved by what produced them and the exact text embedded.

The free tier allows 1000 embedded texts a day and a full rebuild needs ~812,
so a failed run must not throw away what it already paid for, and unchanged
text should never be embedded twice. Vectors are written as each batch comes
back; the next run embeds only what is missing.

Kept apart from the Chroma index on purpose: the index collection is reset on
every rebuild, and a Chroma collection rejects vectors of mixed dimensions.
SQLite (stdlib) stores any vector size under any key.
"""

from __future__ import annotations

import sqlite3
from array import array
from collections.abc import Iterable, Sequence
from contextlib import closing
from pathlib import Path

# SQLite builds before 3.32 cap bound parameters at 999.
_LOOKUP_CHUNK = 500


def cache_key(namespace: str, content_hash: str) -> str:
    """`namespace` identifies the vector space (model, dimensions, task type)."""
    return f"{namespace}|{content_hash}"


def _encode(vector: Sequence[float]) -> bytes:
    # float64, so vectors round-trip exactly.
    return array("d", vector).tobytes()


def _decode(blob: bytes) -> list[float]:
    values = array("d")
    values.frombytes(blob)
    return values.tolist()


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            )

    @classmethod
    def open_existing(cls, path: Path) -> EmbeddingCache | None:
        """The cache at `path`, or None - without creating anything - if there isn't one."""
        return cls(path) if Path(path).is_file() else None

    def _connect(self) -> sqlite3.Connection:
        # One short-lived connection per operation: nothing holds the file open
        # between calls, which matters for Windows file locking.
        return sqlite3.connect(self.path)

    def get_many(self, keys: Iterable[str]) -> dict[str, list[float]]:
        wanted = list(dict.fromkeys(keys))
        found: dict[str, list[float]] = {}
        with closing(self._connect()) as conn:
            for start in range(0, len(wanted), _LOOKUP_CHUNK):
                part = wanted[start : start + _LOOKUP_CHUNK]
                placeholders = ",".join("?" * len(part))
                rows = conn.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})", part
                )
                found.update((key, _decode(blob)) for key, blob in rows)
        return found

    def put_many(self, items: Iterable[tuple[str, Sequence[float]]]) -> None:
        rows = [(key, _encode(vector)) for key, vector in items]
        if not rows:
            return
        with closing(self._connect()) as conn, conn:
            conn.executemany("INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)", rows)

    def count(self) -> int:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
