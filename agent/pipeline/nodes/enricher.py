"""Enricher node: gathers context from multiple providers in parallel with degradation."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from agent.pipeline.state import PipelineState
from agent.providers.base import CodeProvider, DocProvider

log = structlog.get_logger()


async def enricher_node(
    state: PipelineState,
    *,
    code_provider: CodeProvider,
    doc_provider: DocProvider,
) -> PipelineState:
    """Gather code + doc context in parallel. Gracefully degrade on failures."""
    task_raw = state.get("task_raw", {})
    query = task_raw.get("title", "") or task_raw.get("free_text", "")

    warnings: list[str] = []
    enriched: dict[str, Any] = {}

    code_coro = _safe_search(code_provider.search_code(query), "code")
    doc_coro = _safe_search(doc_provider.search_docs(query), "doc")

    code_result, doc_result = await asyncio.gather(code_coro, doc_coro)

    has_code = False
    if isinstance(code_result, list) and len(code_result) > 0:
        enriched["code_snippets"] = code_result
        has_code = True
    elif isinstance(code_result, str):
        warnings.append(code_result)

    has_doc = False
    if isinstance(doc_result, list) and len(doc_result) > 0:
        enriched["doc_pages"] = doc_result
        has_doc = True
    elif isinstance(doc_result, str):
        warnings.append(doc_result)

    log.info(
        "enricher_done",
        has_code=has_code,
        has_doc=has_doc,
        warnings=len(warnings),
    )
    return {
        **state,
        "enriched_context": enriched,
        "has_code_context": has_code,
        "has_doc_context": has_doc,
        "enrichment_warnings": warnings,
    }


async def _safe_search(coro: Any, source: str) -> list[Any] | str:
    """Run a search coroutine; return results or an error string."""
    try:
        result: list[Any] = await coro
        return result
    except Exception as exc:
        log.warning("enricher_source_failed", source=source, error=str(exc))
        return f"{source} unavailable: {exc}"
