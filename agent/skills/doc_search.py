"""DocSearch sub-agent: Pydantic AI agent for searching documentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from agent.providers.base import DocProvider


class DocSearchResult(BaseModel):
    """Structured result from documentation search."""

    pages: list[dict[str, Any]] = Field(default_factory=list)
    pages_found: int = 0
    query_used: str = ""


@dataclass
class DocSearchDeps:
    doc_provider: DocProvider


doc_search_agent = Agent(
    "test",
    output_type=DocSearchResult,
    system_prompt=(
        "You are a documentation search assistant. Given a query, "
        "find relevant documentation pages and return structured results."
    ),
    deps_type=DocSearchDeps,
)


@doc_search_agent.tool
async def search_docs(ctx: RunContext[DocSearchDeps], query: str, limit: int = 10) -> str:
    """Search documentation pages."""
    results = await ctx.deps.doc_provider.search_docs(query, limit=limit)
    if not results:
        return "No documentation found matching the query."
    return "\n---\n".join(
        f"Title: {r.get('title', 'unknown')}\n{r.get('content', '')}" for r in results
    )


@doc_search_agent.tool
async def get_page(ctx: RunContext[DocSearchDeps], page_id: str) -> str:
    """Get a specific documentation page."""
    page = await ctx.deps.doc_provider.get_page(page_id)
    return f"Title: {page.get('title', '')}\n{page.get('content', '')}"


async def run_doc_search(doc_provider: DocProvider, query: str) -> DocSearchResult:
    """Execute the doc search agent and return structured results."""
    result = await doc_search_agent.run(
        f"Search documentation for: {query}",
        deps=DocSearchDeps(doc_provider=doc_provider),
    )
    return result.output
