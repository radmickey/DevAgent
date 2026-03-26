"""LangGraph pipeline: state machine with nodes, conditional edges, HITL.

Providers are auto-resolved from ToolCatalog when available, Stub otherwise.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from agent.pipeline.state import PipelineState

MAX_REVIEW_RETRIES = 3
MAX_HITL_RETRIES = 3


def _should_continue_after_review(state: PipelineState) -> str:
    """Conditional edge after reviewer: pass → doc_writer, fail → executor (max 3x)."""
    review = state.get("review_result", {})
    iteration = state.get("iteration_count", 0)

    if review.get("approved", False):
        return "doc_writer"
    if iteration >= MAX_REVIEW_RETRIES:
        return "doc_writer"
    return "executor"


def _should_continue_after_hitl(state: PipelineState) -> str:
    """Conditional edge after HITL: approved → executor, rejected → explainer (max 3x)."""
    if state.get("plan_approved", False):
        return "executor"
    iteration = state.get("iteration_count", 0)
    if iteration >= MAX_HITL_RETRIES:
        return "executor"
    return "explainer"


def _check_dry_run(state: PipelineState) -> str:
    """If --dry-run, stop before executor."""
    if state.get("dry_run", False):
        return END
    return "hitl"


def _resolve_providers(tool_catalog: Any) -> tuple[Any, Any, Any]:
    """Create providers from ToolCatalog if MCP servers are available, Stub otherwise."""
    from agent.providers.code.stub import StubCodeProvider
    from agent.providers.doc.stub import StubDocProvider
    from agent.providers.task.stub import StubTaskProvider

    task_prov: Any = StubTaskProvider()
    code_prov: Any = StubCodeProvider()
    doc_prov: Any = StubDocProvider()

    if tool_catalog is None:
        return task_prov, code_prov, doc_prov

    try:
        from agent.providers.mcp_classifier import ToolStage
        from agent.providers.mcp_providers import (
            MCPCodeProvider,
            MCPDocProvider,
            MCPTaskProvider,
        )

        for ct in tool_catalog.get_all_tools():
            if ToolStage.TASK_MANAGEMENT in ct.stages:
                task_prov = MCPTaskProvider(tool_catalog._client, ct.server)
                break

        for ct in tool_catalog.get_all_tools():
            if ToolStage.CODE_OPERATIONS in ct.stages or ToolStage.CONTEXT_GATHERING in ct.stages:
                code_prov = MCPCodeProvider(tool_catalog._client, ct.server)
                break

        for ct in tool_catalog.get_all_tools():
            if ToolStage.CONTEXT_GATHERING in ct.stages or ToolStage.PLANNING in ct.stages:
                doc_prov = MCPDocProvider(tool_catalog._client, ct.server)
                break
    except Exception:
        pass

    return task_prov, code_prov, doc_prov


def build_pipeline(
    tool_catalog: Any = None,
) -> tuple:  # type: ignore[type-arg]
    """Build and compile the LangGraph pipeline.

    Args:
        tool_catalog: Optional ToolCatalog for MCP tool injection.
            When provided, providers are auto-created from MCP servers.
            When None, Stub providers are used.

    LLM flags are read from Config (env vars / defaults):
        USE_LLM_ENRICHERS  — LLM enricher sub-agents (code/doc/task/diff search)
        USE_LLM_DOC_WRITER — LLM-based documentation generation
        USE_LLM_CLASSIFIER — LLM input classification (task_type, urgency)
        USE_LLM_META_AGENT — LLM-based prompt self-evolution (L2)

    Returns (compiled_graph, config_dict).
    """
    from agent.config import get_config
    from agent.pipeline.nodes.ranker import ranker_node

    cfg = get_config()
    task_prov, code_prov, doc_prov = _resolve_providers(tool_catalog)

    workflow = StateGraph(PipelineState)

    workflow.add_node(
        "input_router",
        _make_input_router_wrapper(use_llm_classifier=cfg.use_llm_classifier),
    )
    workflow.add_node("reader", _make_reader_wrapper(task_prov))
    workflow.add_node(
        "enricher",
        _make_enricher_wrapper(
            code_prov, doc_prov, task_prov, tool_catalog,
            use_llm_enrichers=cfg.use_llm_enrichers,
        ),
    )
    workflow.add_node("ranker", ranker_node)
    workflow.add_node("explainer", _make_explainer_wrapper(tool_catalog))
    workflow.add_node("hitl", _hitl_passthrough)
    workflow.add_node("executor", _make_executor_wrapper(code_prov, tool_catalog))
    workflow.add_node("reviewer", _make_reviewer_wrapper(code_prov, tool_catalog))
    workflow.add_node(
        "doc_writer",
        _make_doc_writer_wrapper(use_llm=cfg.use_llm_doc_writer),
    )

    workflow.set_entry_point("input_router")
    workflow.add_edge("input_router", "reader")
    workflow.add_edge("reader", "enricher")
    workflow.add_edge("enricher", "ranker")
    workflow.add_edge("ranker", "explainer")

    workflow.add_conditional_edges("explainer", _check_dry_run, {"hitl": "hitl", END: END})
    workflow.add_conditional_edges(
        "hitl",
        _should_continue_after_hitl,
        {"executor": "executor", "explainer": "explainer"},
    )
    workflow.add_edge("executor", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        _should_continue_after_review,
        {"doc_writer": "doc_writer", "executor": "executor"},
    )
    workflow.add_edge("doc_writer", END)

    compiled = workflow.compile(interrupt_before=["hitl"])
    config: dict = {"recursion_limit": 50}
    return compiled, config


def _make_input_router_wrapper(*, use_llm_classifier: bool = False) -> Any:
    from agent.pipeline.nodes.input_router import input_router_node

    async def wrapper(state: PipelineState) -> PipelineState:
        return await input_router_node(state, use_llm_classifier=use_llm_classifier)

    return wrapper


def _make_reader_wrapper(task_prov: Any) -> Any:
    from agent.pipeline.nodes.reader import reader_node

    async def wrapper(state: PipelineState) -> PipelineState:
        return await reader_node(state, task_provider=task_prov)

    return wrapper


def _make_enricher_wrapper(
    code_prov: Any,
    doc_prov: Any,
    task_prov: Any,
    tool_catalog: Any = None,
    *,
    use_llm_enrichers: bool = False,
) -> Any:
    from agent.pipeline.nodes.enricher import enricher_node

    async def wrapper(state: PipelineState) -> PipelineState:
        return await enricher_node(
            state,
            code_provider=code_prov,
            doc_provider=doc_prov,
            task_provider=task_prov,
            tool_catalog=tool_catalog,
            use_llm_enrichers=use_llm_enrichers,
        )

    return wrapper


def _make_explainer_wrapper(tool_catalog: Any = None) -> Any:
    from agent.pipeline.models import NodeDeps
    from agent.pipeline.nodes.explainer import explainer_node

    async def wrapper(state: PipelineState) -> PipelineState:
        deps = NodeDeps(
            task_id=state.get("task_id", ""),
            tool_catalog=tool_catalog,
        )
        return await explainer_node(state, deps=deps)

    return wrapper


def _make_executor_wrapper(code_prov: Any, tool_catalog: Any = None) -> Any:
    from agent.memory.effects import SideEffectTracker
    from agent.pipeline.models import NodeDeps
    from agent.pipeline.nodes.executor import executor_node

    _tracker = SideEffectTracker()

    async def wrapper(state: PipelineState) -> PipelineState:
        deps = NodeDeps(
            task_id=state.get("task_id", ""),
            code_provider=code_prov,
            effects_tracker=_tracker,
            tool_catalog=tool_catalog,
        )
        return await executor_node(state, deps=deps)

    return wrapper


def _make_reviewer_wrapper(code_prov: Any, tool_catalog: Any = None) -> Any:
    from agent.pipeline.models import NodeDeps
    from agent.pipeline.nodes.reviewer import reviewer_node

    async def wrapper(state: PipelineState) -> PipelineState:
        deps = NodeDeps(
            task_id=state.get("task_id", ""),
            code_provider=code_prov,
            tool_catalog=tool_catalog,
        )
        return await reviewer_node(state, deps=deps)

    return wrapper


def _make_doc_writer_wrapper(*, use_llm: bool = False) -> Any:
    from agent.pipeline.nodes.doc_writer import doc_writer_node

    async def wrapper(state: PipelineState) -> PipelineState:
        return await doc_writer_node(state, use_llm=use_llm)

    return wrapper


async def _hitl_passthrough(state: PipelineState) -> PipelineState:
    """HITL checkpoint — graph interrupts before this node for human approval."""
    return {**state, "plan_approved": True}
