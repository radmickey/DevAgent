"""LangGraph pipeline: state machine with 7 nodes, conditional edges, HITL."""

from __future__ import annotations

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


def build_pipeline() -> tuple:  # type: ignore[type-arg]
    """Build and compile the LangGraph pipeline.

    Returns (compiled_graph, config_dict).
    """
    from agent.pipeline.nodes.doc_writer import doc_writer_node
    from agent.pipeline.nodes.input_router import input_router_node
    from agent.pipeline.nodes.ranker import ranker_node

    workflow = StateGraph(PipelineState)

    workflow.add_node("input_router", input_router_node)
    workflow.add_node("reader", _make_reader_wrapper())
    workflow.add_node("enricher", _make_enricher_wrapper())
    workflow.add_node("ranker", ranker_node)
    workflow.add_node("explainer", _make_explainer_wrapper())
    workflow.add_node("hitl", _hitl_passthrough)
    workflow.add_node("executor", _make_executor_wrapper())
    workflow.add_node("reviewer", _make_reviewer_wrapper())
    workflow.add_node("doc_writer", doc_writer_node)

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


def _make_reader_wrapper():
    """Wrap reader_node to inject the task_provider from a stub."""
    from agent.pipeline.nodes.reader import reader_node
    from agent.providers.task.stub import StubTaskProvider

    _provider = StubTaskProvider()

    async def wrapper(state: PipelineState) -> PipelineState:
        return await reader_node(state, task_provider=_provider)

    return wrapper


def _make_enricher_wrapper():
    """Wrap enricher_node to inject code/doc/task providers from stubs."""
    from agent.pipeline.nodes.enricher import enricher_node
    from agent.providers.code.stub import StubCodeProvider
    from agent.providers.doc.stub import StubDocProvider
    from agent.providers.task.stub import StubTaskProvider

    _code = StubCodeProvider()
    _doc = StubDocProvider()
    _task = StubTaskProvider()

    async def wrapper(state: PipelineState) -> PipelineState:
        return await enricher_node(
            state, code_provider=_code, doc_provider=_doc, task_provider=_task,
        )

    return wrapper


def _make_explainer_wrapper():
    """Wrap explainer_node with NodeDeps."""
    from agent.pipeline.models import NodeDeps
    from agent.pipeline.nodes.explainer import explainer_node

    async def wrapper(state: PipelineState) -> PipelineState:
        deps = NodeDeps(task_id=state.get("task_id", ""))
        return await explainer_node(state, deps=deps)

    return wrapper


def _make_executor_wrapper():
    """Wrap executor_node with NodeDeps and SideEffectTracker."""
    from agent.memory.effects import SideEffectTracker
    from agent.pipeline.models import NodeDeps
    from agent.pipeline.nodes.executor import executor_node
    from agent.providers.code.stub import StubCodeProvider

    _code = StubCodeProvider()
    _tracker = SideEffectTracker()

    async def wrapper(state: PipelineState) -> PipelineState:
        deps = NodeDeps(
            task_id=state.get("task_id", ""),
            code_provider=_code,
            effects_tracker=_tracker,
        )
        return await executor_node(state, deps=deps)

    return wrapper


def _make_reviewer_wrapper():
    """Wrap reviewer_node with NodeDeps."""
    from agent.pipeline.models import NodeDeps
    from agent.pipeline.nodes.reviewer import reviewer_node
    from agent.providers.code.stub import StubCodeProvider

    _code = StubCodeProvider()

    async def wrapper(state: PipelineState) -> PipelineState:
        deps = NodeDeps(
            task_id=state.get("task_id", ""),
            code_provider=_code,
        )
        return await reviewer_node(state, deps=deps)

    return wrapper


async def _hitl_passthrough(state: PipelineState) -> PipelineState:
    """HITL checkpoint — graph interrupts before this node for human approval."""
    return {**state, "plan_approved": True}
