"""Concurrent task handling: parallel pipeline execution with resource limits.

Manages multiple task pipelines running concurrently with:
- Configurable concurrency limit
- Per-task isolation via separate state
- Shared cost tracking
- Task queue with priority ordering
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

log = structlog.get_logger()


class TaskStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ManagedTask:
    """A task managed by the concurrent executor."""

    task_id: str
    input_data: dict[str, Any]
    priority: int = 0
    status: TaskStatus = TaskStatus.QUEUED
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: str | None = None
    finished_at: str | None = None


class ConcurrentExecutor:
    """Execute multiple task pipelines concurrently."""

    def __init__(self, max_concurrent: int = 3) -> None:
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: dict[str, ManagedTask] = {}
        self._queue: asyncio.PriorityQueue[tuple[int, str]] = asyncio.PriorityQueue()

    async def submit(
        self,
        task_id: str,
        input_data: dict[str, Any],
        priority: int = 0,
    ) -> ManagedTask:
        """Submit a task for concurrent execution."""
        managed = ManagedTask(
            task_id=task_id,
            input_data=input_data,
            priority=priority,
        )
        self._tasks[task_id] = managed
        await self._queue.put((-priority, task_id))
        log.info("task_submitted", task_id=task_id, priority=priority)
        return managed

    async def run_all(
        self, pipeline_fn: Any
    ) -> list[ManagedTask]:
        """Process all queued tasks with concurrency limit."""
        tasks: list[asyncio.Task[None]] = []

        while not self._queue.empty():
            _, task_id = await self._queue.get()
            managed = self._tasks.get(task_id)
            if not managed or managed.status == TaskStatus.CANCELLED:
                continue
            task = asyncio.create_task(self._run_single(task_id, pipeline_fn))
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return list(self._tasks.values())

    async def _run_single(self, task_id: str, pipeline_fn: Any) -> None:
        """Run a single task with semaphore-limited concurrency."""
        managed = self._tasks[task_id]

        async with self._semaphore:
            managed.status = TaskStatus.RUNNING
            managed.started_at = datetime.now(timezone.utc).isoformat()
            log.info("task_started", task_id=task_id)

            try:
                result = await pipeline_fn(managed.input_data)
                managed.result = result
                managed.status = TaskStatus.COMPLETED
                log.info("task_completed", task_id=task_id)
            except Exception as exc:
                managed.error = str(exc)
                managed.status = TaskStatus.FAILED
                log.error("task_failed", task_id=task_id, error=str(exc))
            finally:
                managed.finished_at = datetime.now(timezone.utc).isoformat()

    def cancel(self, task_id: str) -> bool:
        """Cancel a queued (not running) task."""
        managed = self._tasks.get(task_id)
        if managed and managed.status == TaskStatus.QUEUED:
            managed.status = TaskStatus.CANCELLED
            log.info("task_cancelled", task_id=task_id)
            return True
        return False

    def get_status(self, task_id: str) -> ManagedTask | None:
        return self._tasks.get(task_id)

    def get_all_statuses(self) -> list[dict[str, Any]]:
        return [
            {
                "task_id": t.task_id,
                "status": t.status.value,
                "priority": t.priority,
                "error": t.error,
                "created_at": t.created_at,
                "started_at": t.started_at,
                "finished_at": t.finished_at,
            }
            for t in self._tasks.values()
        ]
