"""Stub task provider for testing and development."""

from __future__ import annotations

from typing import Any

from agent.providers.base import TaskProvider


class StubTaskProvider(TaskProvider):
    """In-memory task provider that returns canned data."""

    async def get_task(self, task_id: str) -> dict[str, Any]:
        return {
            "id": task_id,
            "title": f"Stub task {task_id}",
            "description": "Stub task description for testing",
            "comments": [],
            "labels": ["stub"],
            "status": "open",
        }

    async def get_comments(self, task_id: str) -> list[dict[str, Any]]:
        return []

    async def update_status(self, task_id: str, status: str) -> None:
        pass
