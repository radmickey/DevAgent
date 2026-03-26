"""Executor node: executes the plan (stub for Iteration 1)."""

from __future__ import annotations

import structlog

from agent.pipeline.state import PipelineState

log = structlog.get_logger()


async def executor_node(state: PipelineState) -> PipelineState:
    """Execute the approved plan. Full implementation in Iteration 3."""
    log.info("executor_stub", task_id=state.get("task_id", ""))
    return {**state, "code_changes": []}
