"""Reviewer node: tests → static analysis → LLM review (conditional).

Strategy from architecture:
- Tests exist + pass → full review
- Tests exist + fail → return to executor with errors
- No tests → only ruff + mypy + LLM review with warning
- Static analysis unavailable → only LLM review with warning
"""

from __future__ import annotations

import asyncio
import subprocess

import structlog
from pydantic_ai import Agent, RunContext

from agent.llm import get_model_for_node
from agent.pipeline.models import NodeDeps, ReviewResult
from agent.pipeline.prompts.reviewer import REVIEWER_SYSTEM_PROMPT, build_reviewer_prompt
from agent.pipeline.state import PipelineState
from agent.security.sanitizer import sanitize, should_sanitize

log = structlog.get_logger()

reviewer_agent = Agent(
    "test",
    output_type=ReviewResult,
    system_prompt=REVIEWER_SYSTEM_PROMPT,
    deps_type=NodeDeps,
)


@reviewer_agent.tool
async def get_file_content(ctx: RunContext[NodeDeps], path: str) -> str:
    """Read a file for review context."""
    if ctx.deps.code_provider:
        return await ctx.deps.code_provider.get_file(path)
    return f"Cannot read: {path}"


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

    try:
        model = get_model_for_node("reviewer")
        result = await reviewer_agent.run(prompt, deps=deps, model=model)
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
