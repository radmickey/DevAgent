"""Enricher node: gathers context from multiple sources in parallel with graceful degradation.

Uses asyncio.gather(return_exceptions=True) — never plain gather.
Sets has_code_context, has_doc_context, has_similar_tasks, has_diff_context flags.
Also invokes any MCP tools classified as context_gathering via ToolCatalog.

Each sub-agent (code_search, doc_search, task_search, diff_search) uses its
dedicated system prompt from the centralized prompt registry.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from pydantic_ai import Agent

from agent.llm import get_model_for_node
from agent.pipeline.prompts import get_prompt
from agent.pipeline.state import PipelineState
from agent.providers.base import CodeProvider, DocProvider, TaskProvider

log = structlog.get_logger()

ENRICHER_CODE_PROMPT = get_prompt("enricher_code_search")
ENRICHER_DOC_PROMPT = get_prompt("enricher_doc_search")
ENRICHER_TASK_PROMPT = get_prompt("enricher_task_search")
ENRICHER_DIFF_PROMPT = get_prompt("enricher_diff_search")


async def enricher_node(
    state: PipelineState,
    *,
    code_provider: CodeProvider,
    doc_provider: DocProvider,
    task_provider: TaskProvider | None = None,
    vector_memory: Any = None,
    tool_catalog: Any = None,
    use_llm_enrichers: bool = False,
) -> PipelineState:
    """Gather code + doc + task + vector + MCP context in parallel.

    Uses gather(return_exceptions=True) for graceful degradation.
    Each source that fails produces a warning, not a crash.
    If use_llm_enrichers is True, also runs LLM sub-agents with enricher prompts.
    If a ToolCatalog is provided, also calls all context_gathering MCP tools.
    """
    task_raw = state.get("task_raw", {})
    query = (
        task_raw.get("title", "")
        or task_raw.get("free_text", "")
        or task_raw.get("description", "")
    )
    task_id = state.get("task_id", "")

    coros: dict[str, Any] = {
        "code_snippets": code_provider.search_code(query),
        "doc_pages": doc_provider.search_docs(query),
    }

    if task_provider is not None and task_id:
        coros["task_comments"] = task_provider.get_comments(task_id)

    if vector_memory is not None and query:
        coros["similar_tasks"] = _query_vector(vector_memory, task_id, query)

    if tool_catalog is not None and query:
        _add_catalog_coros(coros, tool_catalog, query)

    if use_llm_enrichers and query:
        coros["llm_code_search"] = _run_enricher_agent(ENRICHER_CODE_PROMPT, "code_search", query)
        coros["llm_doc_search"] = _run_enricher_agent(ENRICHER_DOC_PROMPT, "doc_search", query)
        coros["llm_task_search"] = _run_enricher_agent(ENRICHER_TASK_PROMPT, "task_search", query)
        coros["llm_diff_search"] = _run_enricher_agent(ENRICHER_DIFF_PROMPT, "diff_agent", query)

    keys = list(coros.keys())
    results = await asyncio.gather(*coros.values(), return_exceptions=True)

    enriched: dict[str, Any] = {}
    warnings: list[str] = []
    has_code = False
    has_doc = False
    has_similar = False
    has_diff = False

    for key, result in zip(keys, results):
        if isinstance(result, BaseException):
            warning_msg = f"{key} unavailable: {result}"
            warnings.append(warning_msg)
            log.warning("enricher_source_failed", source=key, error=str(result))
            continue

        has_data = False
        if isinstance(result, list) and len(result) > 0:
            enriched[key] = result
            has_data = True
        elif isinstance(result, (str, dict)) and result:
            enriched[key] = result
            has_data = True

        if has_data:
            if key in ("code_snippets", "llm_code_search"):
                has_code = True
            elif key in ("doc_pages", "llm_doc_search"):
                has_doc = True
            elif key in ("similar_tasks", "task_comments", "llm_task_search"):
                has_similar = True
            elif key in ("llm_diff_search",):
                has_diff = True

    log.info(
        "enricher_done",
        has_code=has_code,
        has_doc=has_doc,
        has_similar=has_similar,
        has_diff=has_diff,
        warnings=len(warnings),
        sources_ok=len(enriched),
    )

    return {
        **state,
        "enriched_context": enriched,
        "has_code_context": has_code,
        "has_doc_context": has_doc,
        "has_similar_tasks": has_similar,
        "has_diff_context": has_diff,
        "enrichment_warnings": warnings,
    }


async def _run_enricher_agent(
    system_prompt: str,
    node_name: str,
    query: str,
) -> dict[str, Any]:
    """Run an LLM enricher sub-agent and parse its JSON output."""
    agent: Agent[None, str] = Agent(
        "test",
        output_type=str,
        system_prompt=system_prompt,
    )
    model = get_model_for_node(node_name)
    result = await agent.run(query, model=model)
    parsed: dict[str, Any] = json.loads(result.output)
    return parsed


def _add_catalog_coros(
    coros: dict[str, Any],
    tool_catalog: Any,
    query: str,
) -> None:
    """Add coroutines for all context_gathering MCP tools."""
    from agent.providers.mcp_classifier import ToolStage

    context_tools = tool_catalog.get_tools_for_stage(ToolStage.CONTEXT_GATHERING)
    for ct in context_tools:
        key = f"mcp_{ct.server}_{ct.name}"
        args = _build_search_args(ct.input_schema, query)
        coros[key] = tool_catalog.call_tool_safe(
            ct.server, ct.name, args, default=None,
        )


def _build_search_args(schema: dict[str, Any], query: str) -> dict[str, Any]:
    """Build arguments for a search/context tool from its schema.

    Heuristic: find the most likely query parameter by name.
    """
    props = schema.get("properties", {})
    query_param_names = ["query", "q", "search", "text", "keyword", "term", "input"]
    for name in query_param_names:
        if name in props:
            return {name: query}
    if props:
        first_param = next(iter(props))
        return {first_param: query}
    return {"query": query}


async def _query_vector(vector_memory: Any, task_id: str, query: str) -> list[dict[str, Any]]:
    """Query vector memory for similar tasks (sync call wrapped for gather)."""
    result: list[dict[str, Any]] = vector_memory.query(task_id, query, n_results=5)
    return result
