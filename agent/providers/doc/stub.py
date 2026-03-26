"""Stub doc provider for testing and development."""

from __future__ import annotations

from typing import Any

from agent.providers.base import DocProvider


class StubDocProvider(DocProvider):
    """In-memory doc provider that returns canned data."""

    async def search_docs(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return []

    async def get_page(self, page_id: str) -> dict[str, Any]:
        return {"id": page_id, "title": "Stub page", "content": ""}
