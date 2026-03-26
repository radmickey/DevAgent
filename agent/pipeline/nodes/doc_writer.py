"""DocWriter node: writes documentation after task completion (stub for Iteration 1)."""

from __future__ import annotations

import structlog

from agent.pipeline.state import PipelineState

log = structlog.get_logger()


async def doc_writer_node(state: PipelineState) -> PipelineState:
    """Write documentation. Full implementation in Iteration 4."""
    log.info("doc_writer_stub", task_id=state.get("task_id", ""))
    return state
