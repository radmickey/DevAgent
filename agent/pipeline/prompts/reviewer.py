"""Reviewer prompts."""

from __future__ import annotations

from typing import Any

from agent.pipeline.prompts import get_prompt

REVIEWER_SYSTEM_PROMPT = get_prompt("reviewer")


def build_reviewer_prompt(
    code_changes: list[dict[str, Any]],
    plan: dict[str, Any],
    test_output: str | None = None,
    lint_output: str | None = None,
) -> str:
    """Build the reviewer prompt from code changes and analysis results."""
    parts = [f"## Plan Summary\n{plan.get('summary', 'No summary')}"]

    parts.append("## Code Changes")
    for change in code_changes:
        path = change.get("path", "unknown")
        action = change.get("action", "unknown")
        diff = change.get("diff", change.get("content", ""))[:2000]
        parts.append(f"### {action}: {path}\n```\n{diff}\n```")

    if test_output is not None:
        parts.append(f"## Test Results\n```\n{test_output[:2000]}\n```")

    if lint_output is not None:
        parts.append(f"## Static Analysis\n```\n{lint_output[:2000]}\n```")

    parts.append("## Instructions\nReview and produce findings. Approve or reject.")
    return "\n".join(parts)
