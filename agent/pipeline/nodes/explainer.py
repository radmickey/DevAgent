"""Explainer node: Pydantic AI Agent → ExplainerResult. Adaptive prompt."""

from __future__ import annotations


import structlog
from pydantic_ai import Agent, RunContext

from agent.llm import get_model_for_node
from agent.pipeline.models import ExplainerResult, NodeDeps
from agent.pipeline.prompts.explainer import EXPLAINER_SYSTEM_PROMPT, build_explainer_prompt
from agent.pipeline.state import PipelineState
from agent.security.sanitizer import sanitize, should_sanitize

log = structlog.get_logger()

explainer_agent = Agent(
    "test",
    output_type=ExplainerResult,
    system_prompt=EXPLAINER_SYSTEM_PROMPT,
    deps_type=NodeDeps,
)


@explainer_agent.tool
async def get_historical_patterns(ctx: RunContext[NodeDeps], query: str) -> str:
    """Search longterm memory for historical patterns relevant to the task."""
    context = ctx.deps.context
    patterns = context.get("patterns", [])
    if not patterns:
        return "No historical patterns available."
    return "\n".join(f"- {p}" for p in patterns[:10])


async def explainer_node(state: PipelineState, *, deps: NodeDeps | None = None) -> PipelineState:
    """Generate execution plan using Pydantic AI agent.

    Prompt adapts to has_code_context, has_doc_context, has_similar_tasks flags.
    """
    task_id = state.get("task_id", "")
    prompt = build_explainer_prompt(state)

    if should_sanitize():
        prompt = sanitize(prompt)

    if deps is None:
        deps = NodeDeps(task_id=task_id)

    try:
        model = get_model_for_node("explainer")
        result = await explainer_agent.run(prompt, deps=deps, model=model)
        plan = result.output.model_dump()
        log.info(
            "explainer_done",
            task_id=task_id,
            steps=len(plan.get("steps", [])),
            complexity=plan.get("estimated_complexity"),
        )
    except Exception as exc:
        log.error("explainer_failed", task_id=task_id, error=str(exc))
        plan = {
            "summary": f"Plan generation failed: {exc}",
            "approach": "manual",
            "steps": [],
            "risks": [str(exc)],
            "estimated_complexity": "high",
        }

    return {**state, "plan": plan}
