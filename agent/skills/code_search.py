"""CodeSearch sub-agent: Pydantic AI agent for searching code in the repository."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from agent.providers.base import CodeProvider


class CodeSearchResult(BaseModel):
    """Structured result from code search."""

    snippets: list[dict[str, Any]] = Field(default_factory=list)
    files_found: int = 0
    query_used: str = ""


@dataclass
class CodeSearchDeps:
    code_provider: CodeProvider


code_search_agent = Agent(
    "test",
    output_type=CodeSearchResult,
    system_prompt=(
        "You are a code search assistant. Given a query about code functionality, "
        "use the search tool to find relevant code snippets. Return structured results."
    ),
    deps_type=CodeSearchDeps,
)


@code_search_agent.tool
async def search_code(ctx: RunContext[CodeSearchDeps], query: str, limit: int = 10) -> str:
    """Search code in the repository."""
    results = await ctx.deps.code_provider.search_code(query, limit=limit)
    if not results:
        return "No code found matching the query."
    return "\n---\n".join(
        f"File: {r.get('path', 'unknown')}\n{r.get('content', '')}" for r in results
    )


@code_search_agent.tool
async def get_file_content(ctx: RunContext[CodeSearchDeps], path: str) -> str:
    """Get the full content of a file."""
    return await ctx.deps.code_provider.get_file(path)


async def run_code_search(code_provider: CodeProvider, query: str) -> CodeSearchResult:
    """Execute the code search agent and return structured results."""
    result = await code_search_agent.run(
        f"Search for code related to: {query}",
        deps=CodeSearchDeps(code_provider=code_provider),
    )
    return result.output
