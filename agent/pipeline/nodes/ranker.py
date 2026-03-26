"""Ranker node: ranks enriched context using ChromaDB threshold, zero LLM calls."""

from __future__ import annotations

import structlog

from agent.pipeline.state import PipelineState

log = structlog.get_logger()

RELEVANCE_THRESHOLD = 0.35


async def ranker_node(state: PipelineState) -> PipelineState:
    """Rank context items by relevance. Uses ChromaDB distance threshold (<0.35)."""
    enriched = state.get("enriched_context", {})
    ranked: list[dict] = []

    for source, items in enriched.items():
        if not isinstance(items, list):
            continue
        for item in items:
            score = item.get("score", 1.0) if isinstance(item, dict) else 1.0
            if score < RELEVANCE_THRESHOLD:
                ranked.append(item if isinstance(item, dict) else {"content": str(item)})

    ranked.sort(key=lambda x: x.get("score", 1.0))

    log.info("ranker_done", total_items=len(ranked))
    return {**state, "ranked_context": ranked}
