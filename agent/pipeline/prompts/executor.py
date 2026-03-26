"""Executor prompts."""

from __future__ import annotations

from typing import Any

from agent.pipeline.prompts import get_prompt

EXECUTOR_SYSTEM_PROMPT = get_prompt("executor")


def build_executor_prompt(plan: dict[str, Any], task_raw: dict[str, Any]) -> str:
    """Build the executor prompt from the approved plan."""
    title = task_raw.get("title", task_raw.get("free_text", ""))
    steps = plan.get("steps", [])

    parts = [f"## Task: {title}", f"## Approach: {plan.get('approach', '')}", "## Steps to execute:"]
    for i, step in enumerate(steps, 1):
        desc = step.get("description", "") if isinstance(step, dict) else str(step)
        fp = step.get("file_path", "") if isinstance(step, dict) else ""
        action = step.get("action", "") if isinstance(step, dict) else ""
        parts.append(f"{i}. [{action}] {fp}: {desc}")

    parts.append("\nExecute each step. Return the file changes and commands run.")
    return "\n".join(parts)
