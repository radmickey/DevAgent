"""DiffAgent sub-agent: Pydantic AI agent for analyzing code diffs."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from agent.providers.base import CodeProvider


class DiffAnalysisResult(BaseModel):
    """Structured result from diff analysis."""

    files_changed: list[str] = Field(default_factory=list)
    summary: str = ""
    patterns_detected: list[str] = Field(default_factory=list)
    risk_areas: list[str] = Field(default_factory=list)


@dataclass
class DiffAgentDeps:
    code_provider: CodeProvider


diff_agent = Agent(
    "test",
    output_type=DiffAnalysisResult,
    system_prompt=(
        "You are a diff analysis assistant. Analyze code diffs to identify "
        "patterns, changes, and potential risk areas."
    ),
    deps_type=DiffAgentDeps,
)


@diff_agent.tool
async def get_diff(ctx: RunContext[DiffAgentDeps], base: str, head: str) -> str:
    """Get the diff between two git refs."""
    return await ctx.deps.code_provider.get_diff(base, head)


@diff_agent.tool
async def get_file(ctx: RunContext[DiffAgentDeps], path: str) -> str:
    """Get file content for context."""
    return await ctx.deps.code_provider.get_file(path)


async def run_diff_analysis(
    code_provider: CodeProvider, base: str, head: str
) -> DiffAnalysisResult:
    """Execute the diff agent and return structured results."""
    result = await diff_agent.run(
        f"Analyze the diff between {base} and {head}",
        deps=DiffAgentDeps(code_provider=code_provider),
    )
    return result.output
