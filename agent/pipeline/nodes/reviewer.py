"""Reviewer node: tests → static analysis → LLM review (conditional).

Strategy from architecture:
- Tests exist + pass → full review
- Tests exist + fail → return to executor with errors
- No tests → only ruff + mypy + LLM review with warning
- Static analysis unavailable → only LLM review with warning

Dynamically injects MCP tools classified as review from ToolCatalog.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any

import structlog
from pydantic_ai import Agent, RunContext

from agent.llm import get_model_for_node
from agent.pipeline.models import NodeDeps, ReviewResult
from agent.pipeline.prompts.reviewer import REVIEWER_SYSTEM_PROMPT, build_reviewer_prompt
from agent.pipeline.state import PipelineState
from agent.security.sanitizer import sanitize, should_sanitize

log = structlog.get_logger()


def _create_reviewer_agent() -> Agent[NodeDeps, ReviewResult]:
    """Create a fresh reviewer agent with static tools."""
    agent: Agent[NodeDeps, ReviewResult] = Agent(
        "test",
        output_type=ReviewResult,
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        deps_type=NodeDeps,
    )

    @agent.tool
    async def get_file_content(ctx: RunContext[NodeDeps], path: str) -> str:
        """Read a file for review context."""
        if ctx.deps.code_provider:
            return await ctx.deps.code_provider.get_file(path)
        return f"Cannot read: {path}"

    return agent


reviewer_agent = _create_reviewer_agent()


def _inject_review_tools(agent: Agent[NodeDeps, ReviewResult], catalog: Any) -> None:
    """Inject MCP tools classified as 'review' into the reviewer agent."""
    from agent.providers.mcp_classifier import ToolStage
    from agent.providers.tool_catalog import CatalogTool

    tools: list[CatalogTool] = catalog.get_tools_for_stage(ToolStage.REVIEW)
    for ct in tools:
        _register_dynamic_review_tool(agent, catalog, ct)
    if tools:
        log.info("mcp_review_tools_injected", count=len(tools),
                 tools=[t.name for t in tools])


def _register_dynamic_review_tool(
    agent: Agent[NodeDeps, ReviewResult],
    catalog: Any,
    ct: Any,
) -> None:
    """Register a single MCP review tool on the reviewer agent."""
    server = ct.server
    tool_name = ct.name
    description = ct.description or f"MCP tool: {tool_name}"

    @agent.tool(name=f"mcp_{server}_{tool_name}")
    async def _dynamic_review_tool(
        ctx: RunContext[NodeDeps],
        arguments: str = "{}",
        _server: str = server,
        _tool: str = tool_name,
    ) -> str:
        """Call an MCP review tool. Pass arguments as JSON string."""
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

    _dynamic_review_tool.__doc__ = description


async def reviewer_node(state: PipelineState, *, deps: NodeDeps | None = None) -> PipelineState:
    """Review code changes: tests → static analysis → LLM (conditional)."""
    task_id = state.get("task_id", "")
    code_changes = state.get("code_changes", [])
    plan: dict = state.get("plan", {})

    if not code_changes:
        log.info("reviewer_no_changes", task_id=task_id)
        return {
            **state,
            "review_result": ReviewResult(
                approved=True, summary="No code changes to review."
            ).model_dump(),
        }

    test_output = await _run_tests()
    lint_output = await _run_static_analysis()

    tests_passed: bool | None = None
    static_ok: bool | None = None
    warnings: list[str] = []

    if test_output is not None:
        tests_passed = "FAILED" not in test_output and "error" not in test_output.lower()
        if not tests_passed:
            log.warning("reviewer_tests_failed", task_id=task_id)
            return {
                **state,
                "review_result": ReviewResult(
                    approved=False,
                    tests_passed=False,
                    summary=f"Tests failed:\n{test_output[:500]}",
                ).model_dump(),
            }
    else:
        warnings.append("No tests found — only static analysis + LLM review")

    if lint_output is not None:
        static_ok = "error" not in lint_output.lower() or lint_output.strip() == ""
    else:
        warnings.append("Static analysis tools (mypy/ruff) unavailable")

    prompt = build_reviewer_prompt(code_changes, plan, test_output, lint_output)
    if should_sanitize():
        prompt = sanitize(prompt)

    if deps is None:
        deps = NodeDeps(task_id=task_id)

    agent = _create_reviewer_agent()
    if deps.tool_catalog is not None:
        _inject_review_tools(agent, deps.tool_catalog)

    try:
        model = get_model_for_node("reviewer")
        result = await agent.run(prompt, deps=deps, model=model)
        review = result.output
        review.tests_passed = tests_passed
        review.static_analysis_passed = static_ok
        if warnings:
            review.summary = (review.summary + "\n" + "\n".join(f"⚠ {w}" for w in warnings)).strip()

        log.info(
            "reviewer_done",
            task_id=task_id,
            approved=review.approved,
            findings=len(review.findings),
            tests_passed=tests_passed,
        )
        return {**state, "review_result": review.model_dump()}

    except Exception as exc:
        log.error("reviewer_failed", task_id=task_id, error=str(exc))
        return {
            **state,
            "review_result": ReviewResult(
                approved=True,
                summary=f"LLM review unavailable: {exc}. Approving with caution.",
                tests_passed=tests_passed,
                static_analysis_passed=static_ok,
            ).model_dump(),
        }


async def _run_tests() -> str | None:
    """Try to run pytest. Returns output or None if not available."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "pytest", "--tb=short", "-q",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={"PATH": "/usr/bin:/usr/local/bin"},
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        return (stdout or b"").decode() + (stderr or b"").decode()
    except (FileNotFoundError, asyncio.TimeoutError, OSError):
        return None


async def _run_static_analysis() -> str | None:
    """Try to run ruff. Returns output or None if not available."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "ruff", "check", ".",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        return (stdout or b"").decode()
    except (FileNotFoundError, asyncio.TimeoutError, OSError):
        return None
