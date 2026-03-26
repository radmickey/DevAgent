"""Explainer node: generates an execution plan (stub for Iteration 1)."""

from __future__ import annotations

import structlog

from agent.pipeline.state import PipelineState

log = structlog.get_logger()


async def explainer_node(state: PipelineState) -> PipelineState:
    """Generate execution plan. Full Pydantic AI agent in Iteration 3."""
    log.info("explainer_stub", task_id=state.get("task_id", ""))
    return {**state, "plan": {"summary": "stub plan", "steps": []}}
