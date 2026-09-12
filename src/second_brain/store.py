"""ChromaDB persistence for embedded chunks.

Chunk ids are deterministic, so upsert is the only write verb needed: a
re-index overwrites in place instead of accumulating duplicates.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from .chunking import Chunk

DEFAULT_COLLECTION = "documents"


class ChunkStore:
    def __init__(self, persist_dir: Path, collection_name: str = DEFAULT_COLLECTION) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            # Cosine suits normalised embedding vectors better than Chroma's L2 default.
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Got {len(embeddings)} embeddings for {len(chunks)} chunks - "
                "the counts must match or chunks would be paired with the wrong vectors."
            )
        if not chunks:
            return

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[list(v) for v in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "project": c.project,
                    "rel_path": c.rel_path,
                    "path": c.path,
                    "heading_path": c.heading_path,
                    "chunk_index": c.chunk_index,
                    "content_hash": c.content_hash,
                    "mtime": c.mtime,
                }
                for c in chunks
            ],
        )

    def get(self, chunk_id: str) -> dict[str, Any]:
        result = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not result["ids"]:
            raise KeyError(chunk_id)
        return {"text": result["documents"][0], **result["metadatas"][0]}

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def all_metadata(self) -> list[dict[str, Any]]:
        return list(self._collection.get(include=["metadatas"])["metadatas"])
