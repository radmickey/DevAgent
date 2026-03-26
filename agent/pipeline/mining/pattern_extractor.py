"""Pattern extractor: analyzes completed tasks and extracts reusable patterns.

Uses the pattern_extractor system prompt from the centralized registry.
Input: task metadata, merged PR diff, review comments.
Output: TaskPattern dict with solution_approach, key_patterns, common_mistakes.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from pydantic_ai import Agent

from agent.llm import get_model_for_node
from agent.pipeline.prompts import get_prompt

log = structlog.get_logger()

PATTERN_EXTRACTOR_SYSTEM_PROMPT = get_prompt("pattern_extractor")


async def extract_patterns(
    task: dict[str, Any],
    diff: str,
    review_comments: list[str],
) -> dict[str, Any] | None:
    """Extract reusable patterns from a completed task via LLM.

    Returns a TaskPattern dict or None if extraction fails or
    the task has no meaningful diff.
    """
    if not diff:
        log.info("pattern_extractor_skip", reason="empty diff")
        return None

    user_prompt = _build_prompt(task, diff, review_comments)

    try:
        agent: Agent[None, str] = Agent(
            "test",
            output_type=str,
            system_prompt=PATTERN_EXTRACTOR_SYSTEM_PROMPT,
        )
        model = get_model_for_node("meta_agent")
        result = await agent.run(user_prompt, model=model)
        parsed: dict[str, Any] | None = json.loads(result.output)
        if parsed is None:
            log.info("pattern_extractor_null", reason="LLM returned null")
            return None
        log.info(
            "pattern_extracted",
            task_type=parsed.get("task_type"),
            patterns=len(parsed.get("key_patterns", [])),
        )
        return parsed
    except Exception as exc:
        log.warning("pattern_extractor_failed", error=str(exc))
        return None


def _build_prompt(
    task: dict[str, Any],
    diff: str,
    review_comments: list[str],
) -> str:
    """Build user prompt for pattern extraction."""
    parts = [
        f"## Task\nID: {task.get('id', '?')}\n"
        f"Title: {task.get('title', task.get('free_text', ''))}\n"
        f"Type: {task.get('type', 'unknown')}",
    ]
    parts.append(f"## Diff\n```\n{diff[:5000]}\n```")

    if review_comments:
        comments_str = "\n".join(f"- {c}" for c in review_comments[:20])
        parts.append(f"## Review Comments\n{comments_str}")
    else:
        parts.append("## Review Comments\nNo review comments.")

    return "\n\n".join(parts)
