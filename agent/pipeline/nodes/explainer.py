"""Explainer node: Pydantic AI Agent → ExplainerResult. Adaptive prompt.

Dynamically injects MCP tools classified as planning from ToolCatalog.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic_ai import Agent, RunContext

from agent.llm import get_model_for_node
from agent.pipeline.models import ExplainerResult, NodeDeps
from agent.pipeline.prompts.explainer import EXPLAINER_SYSTEM_PROMPT, build_explainer_prompt
from agent.pipeline.state import PipelineState
from agent.security.sanitizer import sanitize, should_sanitize

log = structlog.get_logger()


def _create_explainer_agent() -> Agent[NodeDeps, ExplainerResult]:
    """Create a fresh explainer agent with static tools."""
    agent: Agent[NodeDeps, ExplainerResult] = Agent(
        "test",
        output_type=ExplainerResult,
        system_prompt=EXPLAINER_SYSTEM_PROMPT,
        deps_type=NodeDeps,
    )

    @agent.tool
    async def get_historical_patterns(ctx: RunContext[NodeDeps], query: str) -> str:
        """Search longterm memory for historical patterns relevant to the task."""
        context = ctx.deps.context
        patterns = context.get("patterns", [])
        if not patterns:
            return "No historical patterns available."
        return "\n".join(f"- {p}" for p in patterns[:10])

    return agent


explainer_agent = _create_explainer_agent()


def _inject_planning_tools(agent: Agent[NodeDeps, ExplainerResult], catalog: Any) -> None:
    """Inject MCP tools classified as 'planning' into the explainer agent."""
    from agent.providers.mcp_classifier import ToolStage
    from agent.providers.tool_catalog import CatalogTool

    tools: list[CatalogTool] = catalog.get_tools_for_stage(ToolStage.PLANNING)
    for ct in tools:
        _register_dynamic_planning_tool(agent, catalog, ct)
    if tools:
        log.info("mcp_planning_tools_injected", count=len(tools),
                 tools=[t.name for t in tools])


def _register_dynamic_planning_tool(
    agent: Agent[NodeDeps, ExplainerResult],
    catalog: Any,
    ct: Any,
) -> None:
    """Register a single MCP planning tool on the explainer agent."""
    server = ct.server
    tool_name = ct.name
    description = ct.description or f"MCP tool: {tool_name}"

    @agent.tool(name=f"mcp_{server}_{tool_name}")
    async def _dynamic_planning_tool(
        ctx: RunContext[NodeDeps],
        arguments: str = "{}",
        _server: str = server,
        _tool: str = tool_name,
    ) -> str:
        """Call an MCP planning tool. Pass arguments as JSON string."""
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

    _dynamic_planning_tool.__doc__ = description


async def explainer_node(state: PipelineState, *, deps: NodeDeps | None = None) -> PipelineState:
    """Generate execution plan using Pydantic AI agent.

    Prompt adapts to has_code_context, has_doc_context, has_similar_tasks flags.
    Dynamically injects MCP planning tools if a ToolCatalog is available.
    """
    task_id = state.get("task_id", "")
    prompt = build_explainer_prompt(state)

    if should_sanitize():
        prompt = sanitize(prompt)

    if deps is None:
        deps = NodeDeps(task_id=task_id)

    agent = _create_explainer_agent()
    if deps.tool_catalog is not None:
        _inject_planning_tools(agent, deps.tool_catalog)

    try:
        model = get_model_for_node("explainer")
        result = await agent.run(prompt, deps=deps, model=model)
        plan = result.output.model_dump()
        log.info(
            "explainer_done",
            task_id=task_id,
            steps=len(plan.get("steps", [])),
            complexity=plan.get("estimated_complexity"),
        )
    except Exception as exc:
        log.error("explainer_failed", task_id=task_id, error=str(exc))
        plan = {
            "summary": f"Plan generation failed: {exc}",
            "approach": "manual",
            "steps": [],
            "risks": [str(exc)],
            "estimated_complexity": "high",
        }

    return {**state, "plan": plan}
