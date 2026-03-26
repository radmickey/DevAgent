"""ChromaDB vector memory: store, query, TTL cleanup."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import chromadb
import structlog

from agent.config import DEVAGENT_HOME

log = structlog.get_logger()

_CHROMA_PATH = DEVAGENT_HOME / "chroma"


class VectorMemory:
    """Local ChromaDB-backed vector memory for context retrieval."""

    def __init__(self, persist_dir: str | None = None) -> None:
        path = persist_dir or str(_CHROMA_PATH)
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name="devagent_context",
            metadata={"hnsw:space": "cosine"},
        )

    def store(self, task_id: str, items: list[dict[str, Any]]) -> None:
        """Store context items for a task."""
        if not items:
            return

        ids = [f"{task_id}_{i}" for i in range(len(items))]
        documents = [item.get("content", str(item)) for item in items]
        now_ts = datetime.now(timezone.utc).timestamp()
        metadatas: list[dict[str, str | int | float | bool]] = [
            {
                "task_id": task_id,
                "source": item.get("source", "unknown"),
                "created_at": now_ts,
            }
            for item in items
        ]

        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)  # type: ignore[arg-type]
        log.info("vector_stored", task_id=task_id, count=len(items))

    def query(self, task_id: str, query_text: str, max_tokens: int = 500, n_results: int = 5) -> list[dict[str, Any]]:
        """Query similar context. Returns items with score < threshold."""
        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self._collection.count() or 1),
        )

        items = []
        if results and results["documents"]:
            docs = results["documents"][0]
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)  # type: ignore[index]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)  # type: ignore[index]

            for doc, dist, meta in zip(docs, distances, metas):
                items.append({
                    "content": doc,
                    "score": dist,
                    "source": meta.get("source", "unknown"),
                    "task_id": meta.get("task_id", ""),
                })

        return items

    def cleanup_old(self, days: int = 30) -> int:
        """Delete embeddings older than N days. Returns count deleted."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
        old_items = self._collection.get(
            where={"created_at": {"$lt": cutoff}},  # type: ignore[dict-item]
        )
        if old_items and old_items["ids"]:
            self._collection.delete(ids=old_items["ids"])
            count = len(old_items["ids"])
            log.info("vector_cleanup", deleted=count, days=days)
            return count
        return 0
