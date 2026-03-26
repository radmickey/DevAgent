"""Input Router node: determines input type (task_id / text / both) and normalizes."""

from __future__ import annotations

import re

import structlog

from agent.pipeline.state import PipelineState

log = structlog.get_logger()


def _looks_like_task_id(text: str) -> bool:
    """Check if input looks like a tracker task ID (e.g. PROJ-123, #456)."""
    return bool(re.match(r"^[A-Z]+-\d+$", text.strip())) or bool(
        re.match(r"^#?\d+$", text.strip())
    )


async def input_router_node(state: PipelineState) -> PipelineState:
    """Parse raw input, detect type, populate task_id in state."""
    task_id = state.get("task_id", "")

    if not task_id:
        log.error("input_router_no_input")
        return {**state, "task_id": "", "task_raw": {"error": "no input provided"}}

    task_id = task_id.strip()

    if _looks_like_task_id(task_id):
        log.info("input_router_task_id", task_id=task_id)
        return {**state, "task_id": task_id, "task_raw": {}}
    else:
        log.info("input_router_free_text", text=task_id[:80])
        return {
            **state,
            "task_id": "",
            "task_raw": {"free_text": task_id},
        }
