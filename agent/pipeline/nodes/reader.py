"""Reader node: fetches task data from the task provider."""

from __future__ import annotations

from typing import Any

import structlog

from agent.config import TIMEOUTS
from agent.errors import DegradedError, PermanentError, TransientError
from agent.pipeline.state import PipelineState
from agent.providers.base import TaskProvider
from agent.utils.timeout import with_timeout

log = structlog.get_logger()


async def reader_node(
    state: PipelineState,
    *,
    task_provider: TaskProvider,
) -> PipelineState:
    """Read task data from the tracker. Static tools: get_task, get_comments."""
    task_id = state.get("task_id", "")

    if not task_id:
        log.warning("reader_skip_no_task_id")
        return state

    try:
        task_data: dict[str, Any] = await with_timeout(
            task_provider.get_task(task_id),
            timeout=TIMEOUTS["mcp_read"],
            name=f"TaskProvider.get_task({task_id})",
        )
    except TransientError:
        log.warning("reader_timeout", task_id=task_id)
        raise
    except Exception as exc:
        raise PermanentError(f"Failed to read task {task_id}: {exc}") from exc

    try:
        comments = await with_timeout(
            task_provider.get_comments(task_id),
            timeout=TIMEOUTS["mcp_read"],
            name=f"TaskProvider.get_comments({task_id})",
        )
        task_data["comments"] = comments
    except (TransientError, DegradedError):
        log.warning("reader_comments_unavailable", task_id=task_id)
        task_data["comments"] = []

    log.info("reader_done", task_id=task_id, title=task_data.get("title", ""))
    return {**state, "task_raw": task_data}
