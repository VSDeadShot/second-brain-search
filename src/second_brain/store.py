"""ChromaDB persistence for embedded chunks.

Chunk ids are deterministic, so upsert is the only write verb needed: a
re-index overwrites in place instead of accumulating duplicates.

The collection records which vector space built it (`embedding_namespace`:
model, dimensions, task type). Chroma only catches a dimension mismatch; a
different model at the same 768 dimensions would be searched without complaint
and return confidently wrong neighbours. Recording the space lets retrieval
and incremental writes refuse that.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from .chunking import Chunk

DEFAULT_COLLECTION = "documents"
_NAMESPACE_KEY = "embedding_namespace"


class VectorSpaceMismatch(Exception):
    """Vectors from one embedding space were about to be mixed with another's."""


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

    @property
    def namespace(self) -> str | None:
        """The vector space this index was built in, or None if never recorded."""
        return (self._collection.metadata or {}).get(_NAMESPACE_KEY)

    def record_namespace(self, namespace: str) -> None:
        # Only the namespace key is passed: re-sending "hnsw:space" makes Chroma
        # raise ("changing the distance function ... is not supported"). Probed on
        # chromadb 1.5.9 - the collection stays cosine even though the key then
        # disappears from `metadata`; a test pins that.
        self._collection.modify(metadata={_NAMESPACE_KEY: namespace})

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

    def query(
        self, vector: Sequence[float], *, k: int, project: str | None = None
    ) -> list[dict[str, Any]]:
        """Nearest chunks first: each row has id, text, distance and all metadata."""
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=k,
            where={"project": project} if project is not None else None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"id": chunk_id, "text": text, "distance": distance, **metadata}
            for chunk_id, text, distance, metadata in zip(
                result["ids"][0],
                result["documents"][0],
                result["distances"][0],
                result["metadatas"][0],
            )
        ]

    def get(self, chunk_id: str) -> dict[str, Any]:
        result = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not result["ids"]:
            raise KeyError(chunk_id)
        return {"text": result["documents"][0], **result["metadatas"][0]}

    def count(self) -> int:
        return self._collection.count()

    def projects(self) -> list[str]:
        return sorted({m["project"] for m in self.all_metadata()}, key=str.lower)

    def reset(self, *, namespace: str | None = None) -> None:
        self._client.delete_collection(self.collection_name)
        metadata: dict[str, Any] = {"hnsw:space": "cosine"}
        if namespace is not None:
            metadata[_NAMESPACE_KEY] = namespace
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata=metadata
        )

    def all_ids(self) -> list[str]:
        return list(self._collection.get(include=[])["ids"])

    def all_metadata(self) -> list[dict[str, Any]]:
        return list(self._collection.get(include=["metadatas"])["metadatas"])
