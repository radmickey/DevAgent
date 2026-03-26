"""Stub code provider for testing and development."""

from __future__ import annotations

from typing import Any

from agent.providers.base import CodeProvider


class StubCodeProvider(CodeProvider):
    """In-memory code provider that returns canned data."""

    async def search_code(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return []

    async def get_file(self, path: str, ref: str = "main") -> str:
        return ""

    async def list_files(self, path: str = "", ref: str = "main") -> list[str]:
        return []

    async def get_diff(self, base: str, head: str) -> str:
        return ""
