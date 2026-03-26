"""Executor node: Pydantic AI Agent → ExecutorResult. Side effects tracking.

Dynamically injects MCP tools classified as code_operations from ToolCatalog.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic_ai import Agent, RunContext

from agent.llm import get_model_for_node
from agent.memory.effects import SideEffectTracker
from agent.pipeline.models import ExecutorResult, NodeDeps
from agent.pipeline.prompts.executor import EXECUTOR_SYSTEM_PROMPT, build_executor_prompt
from agent.pipeline.state import PipelineState
from agent.security.sanitizer import sanitize, should_sanitize

log = structlog.get_logger()


def _create_executor_agent() -> Agent[NodeDeps, ExecutorResult]:
    """Create a fresh executor agent with static tools."""
    agent: Agent[NodeDeps, ExecutorResult] = Agent(
        "test",
        output_type=ExecutorResult,
        system_prompt=EXECUTOR_SYSTEM_PROMPT,
        deps_type=NodeDeps,
    )

    @agent.tool
    async def read_file(ctx: RunContext[NodeDeps], path: str) -> str:
        """Read a file from the repository."""
        if ctx.deps.code_provider:
            return await ctx.deps.code_provider.get_file(path)
        return f"File not accessible: {path}"

    @agent.tool
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

    return agent


executor_agent = _create_executor_agent()


def inject_mcp_tools(agent: Agent[NodeDeps, ExecutorResult], catalog: Any, stage: str) -> None:
    """Dynamically inject MCP tools from ToolCatalog into a Pydantic AI agent."""
    from agent.providers.tool_catalog import CatalogTool

    tools: list[CatalogTool] = catalog.get_tools_for_stage(stage)
    for ct in tools:
        _register_dynamic_tool(agent, catalog, ct)

    if tools:
        log.info("mcp_tools_injected", stage=stage, count=len(tools),
                 tools=[t.name for t in tools])


def _register_dynamic_tool(
    agent: Agent[NodeDeps, ExecutorResult],
    catalog: Any,
    ct: Any,
) -> None:
    """Register a single MCP tool as a Pydantic AI tool on the agent."""
    server = ct.server
    tool_name = ct.name
    description = ct.description or f"MCP tool: {tool_name}"

    @agent.tool(name=f"mcp_{server}_{tool_name}")
    async def _dynamic_mcp_tool(
        ctx: RunContext[NodeDeps],
        arguments: str = "{}",
        _server: str = server,
        _tool: str = tool_name,
    ) -> str:
        """Call an MCP tool. Pass arguments as a JSON string."""
        import json
        try:
            args = json.loads(arguments) if arguments else {}
        except (json.JSONDecodeError, ValueError):
            args = {"input": arguments}

        cat = ctx.deps.tool_catalog
        if cat is None:
            return f"Tool catalog not available for {_tool}"

        result = await cat.call_tool_safe(_server, _tool, args, default=None)
        if result is None:
            return f"Tool {_tool} returned no result."
        return str(result)[:2000]

    _dynamic_mcp_tool.__doc__ = description


async def executor_node(state: PipelineState, *, deps: NodeDeps | None = None) -> PipelineState:
    """Execute the approved plan using Pydantic AI agent.

    Records all side effects for potential rollback.
    Dynamically injects MCP tools if a ToolCatalog is available.
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

    agent = _create_executor_agent()
    if deps.tool_catalog is not None:
        inject_mcp_tools(agent, deps.tool_catalog, "code_operations")

    effects_tracker = deps.effects_tracker

    try:
        model = get_model_for_node("executor")
        result = await agent.run(prompt, deps=deps, model=model)
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
