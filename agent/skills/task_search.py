"""TaskSearch sub-agent: Pydantic AI agent for finding related tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from agent.providers.base import TaskProvider


class TaskSearchResult(BaseModel):
    """Structured result from task search."""

    related_tasks: list[dict[str, Any]] = Field(default_factory=list)
    tasks_found: int = 0
    query_used: str = ""


@dataclass
class TaskSearchDeps:
    task_provider: TaskProvider


task_search_agent = Agent(
    "test",
    output_type=TaskSearchResult,
    system_prompt=(
        "You are a task search assistant. Given a task description, "
        "find related or similar tasks from the tracker."
    ),
    deps_type=TaskSearchDeps,
)


@task_search_agent.tool
async def get_task_details(ctx: RunContext[TaskSearchDeps], task_id: str) -> str:
    """Get details of a specific task."""
    task = await ctx.deps.task_provider.get_task(task_id)
    return (
        f"ID: {task.get('id', '')}\n"
        f"Title: {task.get('title', '')}\n"
        f"Status: {task.get('status', '')}\n"
        f"Description: {task.get('description', '')}"
    )


@task_search_agent.tool
async def get_task_comments(ctx: RunContext[TaskSearchDeps], task_id: str) -> str:
    """Get comments on a task."""
    comments = await ctx.deps.task_provider.get_comments(task_id)
    if not comments:
        return "No comments on this task."
    return "\n---\n".join(
        f"Author: {c.get('author', 'unknown')}: {c.get('text', '')}" for c in comments
    )


async def run_task_search(task_provider: TaskProvider, query: str) -> TaskSearchResult:
    """Execute the task search agent and return structured results."""
    result = await task_search_agent.run(
        f"Find tasks related to: {query}",
        deps=TaskSearchDeps(task_provider=task_provider),
    )
    return result.output
