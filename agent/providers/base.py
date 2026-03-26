"""Abstract provider interfaces: TaskProvider, CodeProvider, DocProvider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TaskProvider(ABC):
    """Reads task/issue data from a tracker (Jira, Tracker, Linear)."""

    @abstractmethod
    async def get_task(self, task_id: str) -> dict[str, Any]:
        """Fetch a single task by ID."""

    @abstractmethod
    async def get_comments(self, task_id: str) -> list[dict[str, Any]]:
        """Fetch comments for a task."""

    @abstractmethod
    async def update_status(self, task_id: str, status: str) -> None:
        """Update task status."""


class CodeProvider(ABC):
    """Reads code from a repository (GitHub, Arcanum, GitLab)."""

    @abstractmethod
    async def search_code(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search code in the repository."""

    @abstractmethod
    async def get_file(self, path: str, ref: str = "main") -> str:
        """Get file content by path."""

    @abstractmethod
    async def list_files(self, path: str = "", ref: str = "main") -> list[str]:
        """List files in a directory."""

    @abstractmethod
    async def get_diff(self, base: str, head: str) -> str:
        """Get diff between two refs."""


class DocProvider(ABC):
    """Reads documentation (Notion, Confluence, Wiki)."""

    @abstractmethod
    async def search_docs(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search documentation pages."""

    @abstractmethod
    async def get_page(self, page_id: str) -> dict[str, Any]:
        """Get a single documentation page."""
