"""Executor node: Pydantic AI Agent → ExecutorResult. Side effects tracking."""

from __future__ import annotations


import structlog
from pydantic_ai import Agent, RunContext

from agent.llm import get_model_for_node
from agent.memory.effects import SideEffectTracker
from agent.pipeline.models import ExecutorResult, NodeDeps
from agent.pipeline.prompts.executor import EXECUTOR_SYSTEM_PROMPT, build_executor_prompt
from agent.pipeline.state import PipelineState
from agent.security.sanitizer import sanitize, should_sanitize

log = structlog.get_logger()

executor_agent = Agent(
    "test",
    output_type=ExecutorResult,
    system_prompt=EXECUTOR_SYSTEM_PROMPT,
    deps_type=NodeDeps,
)


@executor_agent.tool
async def read_file(ctx: RunContext[NodeDeps], path: str) -> str:
    """Read a file from the repository."""
    if ctx.deps.code_provider:
        return await ctx.deps.code_provider.get_file(path)
    return f"File not accessible: {path}"


@executor_agent.tool
async def search_code(ctx: RunContext[NodeDeps], query: str) -> str:
    """Search code in the repository for reference."""
    if ctx.deps.code_provider:
        results = await ctx.deps.code_provider.search_code(query, limit=5)
        if not results:
            return "No results found."
        return "\n---\n".join(
            f"{r.get('path', '?')}: {str(r.get('content', ''))[:300]}" for r in results
        )
    return "Code provider not available."


async def executor_node(state: PipelineState, *, deps: NodeDeps | None = None) -> PipelineState:
    """Execute the approved plan using Pydantic AI agent.

    Records all side effects for potential rollback.
    """
    task_id = state.get("task_id", "")
    plan: dict = state.get("plan", {})
    task_raw = state.get("task_raw", {})

    if not plan or not plan.get("steps"):
        log.warning("executor_no_plan", task_id=task_id)
        return {**state, "code_changes": []}

    prompt = build_executor_prompt(plan, task_raw)
    if should_sanitize():
        prompt = sanitize(prompt)

    if deps is None:
        deps = NodeDeps(task_id=task_id)

    effects_tracker = deps.effects_tracker

    try:
        model = get_model_for_node("executor")
        result = await executor_agent.run(prompt, deps=deps, model=model)
        executor_result = result.output

        changes = [c.model_dump() for c in executor_result.files_changed]

        if effects_tracker and isinstance(effects_tracker, SideEffectTracker):
            for change in executor_result.files_changed:
                effects_tracker.record(
                    task_id, "file_written", {"path": change.path, "action": change.action}
                )
            for cmd in executor_result.commands_run:
                effects_tracker.record(task_id, "command_run", {"command": cmd})

        log.info(
            "executor_done",
            task_id=task_id,
            files_changed=len(changes),
            commands_run=len(executor_result.commands_run),
        )
    except Exception as exc:
        log.error("executor_failed", task_id=task_id, error=str(exc))
        changes = []

    iteration = state.get("iteration_count", 0) + 1
    return {**state, "code_changes": changes, "iteration_count": iteration}
