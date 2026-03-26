"""Executor prompts."""

from __future__ import annotations

from typing import Any

EXECUTOR_SYSTEM_PROMPT = """\
You are a senior software engineer executing a plan. For each step:
1. Write the actual code changes needed
2. Specify exact file paths and actions (create/modify/delete)
3. Track all side effects (branches, commits, file writes)
4. If a step cannot be completed, explain why

Be precise. Write production-quality code.
"""


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
