"""Reviewer node: reviews code changes (stub for Iteration 1)."""

from __future__ import annotations

import structlog

from agent.pipeline.state import PipelineState

log = structlog.get_logger()


async def reviewer_node(state: PipelineState) -> PipelineState:
    """Review code changes. Full implementation in Iteration 3."""
    log.info("reviewer_stub", task_id=state.get("task_id", ""))
    return {**state, "review_result": {"approved": True, "findings": []}}
