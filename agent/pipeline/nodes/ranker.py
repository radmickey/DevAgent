"""Ranker node: ranks enriched context using ChromaDB distance threshold.

Key rules from architecture:
- ChromaDB score threshold < 0.35
- Zero LLM calls — ranking is purely distance-based
- Items above threshold are discarded
"""

from __future__ import annotations

from typing import Any

import structlog

from agent.memory.budget import fit_context
from agent.pipeline.state import PipelineState

log = structlog.get_logger()

RELEVANCE_THRESHOLD = 0.35


async def ranker_node(
    state: PipelineState,
    *,
    vector_memory: Any = None,
    max_context_tokens: int = 8_000,
) -> PipelineState:
    """Rank context items by relevance score, apply threshold, fit to budget.

    1. Collect all items from enriched_context
    2. If vector_memory available, score items via ChromaDB distance
    3. Filter by threshold (< 0.35)
    4. Sort by score ascending (lower = more relevant in cosine distance)
    5. Fit to token budget
    """
    enriched = state.get("enriched_context", {})
    task_raw = state.get("task_raw", {})
    query = task_raw.get("title", "") or task_raw.get("free_text", "") or ""

    all_items: list[dict[str, Any]] = []

    for source, items in enriched.items():
        if not isinstance(items, list):
            continue
        for item in items:
            entry = item if isinstance(item, dict) else {"content": str(item)}
            entry.setdefault("source", source)
            entry.setdefault("score", 1.0)
            all_items.append(entry)

    if vector_memory is not None and query and all_items:
        all_items = _score_via_vector(vector_memory, query, all_items)

    ranked = [item for item in all_items if item.get("score", 1.0) < RELEVANCE_THRESHOLD]
    ranked.sort(key=lambda x: x.get("score", 1.0))

    if not ranked and all_items:
        ranked = sorted(all_items, key=lambda x: x.get("score", 1.0))[:10]
        log.info("ranker_fallback", reason="no items below threshold", kept=len(ranked))

    budget_items = [{"content": _item_text(r), "relevance": 1.0 - r.get("score", 0.5)} for r in ranked]
    fitted = fit_context(budget_items, max_tokens=max_context_tokens)

    final_ranked = ranked[: len(fitted)]

    log.info(
        "ranker_done",
        total_input=len(all_items),
        below_threshold=len([i for i in all_items if i.get("score", 1.0) < RELEVANCE_THRESHOLD]),
        final_count=len(final_ranked),
    )
    return {**state, "ranked_context": final_ranked}


def _score_via_vector(vector_memory: Any, query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-score items using ChromaDB's embedding distance."""
    try:
        results = vector_memory.query("", query, n_results=20)
        scored_contents: dict[str, float] = {}
        for r in results:
            scored_contents[r.get("content", "")] = r.get("score", 1.0)

        for item in items:
            content = _item_text(item)
            if content in scored_contents:
                item["score"] = scored_contents[content]
    except Exception as exc:
        log.warning("ranker_vector_scoring_failed", error=str(exc))

    return items


def _item_text(item: dict[str, Any]) -> str:
    """Extract displayable text from an item dict."""
    return str(item.get("content", item.get("text", str(item))))
