"""Enricher node: gathers context from multiple sources in parallel with graceful degradation.

Uses asyncio.gather(return_exceptions=True) — never plain gather.
Sets has_code_context, has_doc_context, has_similar_tasks flags.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from agent.pipeline.state import PipelineState
from agent.providers.base import CodeProvider, DocProvider, TaskProvider

log = structlog.get_logger()


async def enricher_node(
    state: PipelineState,
    *,
    code_provider: CodeProvider,
    doc_provider: DocProvider,
    task_provider: TaskProvider | None = None,
    vector_memory: Any = None,
) -> PipelineState:
    """Gather code + doc + task + vector context in parallel.

    Uses gather(return_exceptions=True) for graceful degradation.
    Each source that fails produces a warning, not a crash.
    """
    task_raw = state.get("task_raw", {})
    query = task_raw.get("title", "") or task_raw.get("free_text", "") or task_raw.get("description", "")
    task_id = state.get("task_id", "")

    coros: dict[str, Any] = {
        "code_snippets": code_provider.search_code(query),
        "doc_pages": doc_provider.search_docs(query),
    }

    if task_provider is not None and task_id:
        coros["task_comments"] = task_provider.get_comments(task_id)

    if vector_memory is not None and query:
        coros["similar_tasks"] = _query_vector(vector_memory, task_id, query)

    keys = list(coros.keys())
    results = await asyncio.gather(*coros.values(), return_exceptions=True)

    enriched: dict[str, Any] = {}
    warnings: list[str] = []
    has_code = False
    has_doc = False
    has_similar = False

    for key, result in zip(keys, results):
        if isinstance(result, BaseException):
            warning_msg = f"{key} unavailable: {result}"
            warnings.append(warning_msg)
            log.warning("enricher_source_failed", source=key, error=str(result))
            continue

        if isinstance(result, list) and len(result) > 0:
            enriched[key] = result
            if key == "code_snippets":
                has_code = True
            elif key == "doc_pages":
                has_doc = True
            elif key == "similar_tasks":
                has_similar = True

    log.info(
        "enricher_done",
        has_code=has_code,
        has_doc=has_doc,
        has_similar=has_similar,
        warnings=len(warnings),
        sources_ok=len(enriched),
    )

    return {
        **state,
        "enriched_context": enriched,
        "has_code_context": has_code,
        "has_doc_context": has_doc,
        "has_similar_tasks": has_similar,
        "enrichment_warnings": warnings,
    }


async def _query_vector(vector_memory: Any, task_id: str, query: str) -> list[dict[str, Any]]:
    """Query vector memory for similar tasks (sync call wrapped for gather)."""
    result: list[dict[str, Any]] = vector_memory.query(task_id, query, n_results=5)
    return result
