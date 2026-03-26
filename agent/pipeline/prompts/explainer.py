"""Explainer prompts: adaptive based on has_*_context flags."""

from __future__ import annotations

from agent.pipeline.state import PipelineState

EXPLAINER_SYSTEM_PROMPT = """\
You are an expert software architect. Given a task and its context, produce \
a detailed execution plan with concrete steps.

Rules:
- Each step must specify: description, file_path (if applicable), action (create/modify/delete/run), rationale
- Estimate complexity as low/medium/high
- List risks that could block execution
- Be specific about file paths and what changes are needed
- If context is limited, note what assumptions you're making
"""


def build_explainer_prompt(state: PipelineState) -> str:
    """Build the user prompt for the Explainer, adapting to available context."""
    task_raw = state.get("task_raw", {})
    ranked = state.get("ranked_context", [])
    has_code = state.get("has_code_context", False)
    has_doc = state.get("has_doc_context", False)
    has_similar = state.get("has_similar_tasks", False)
    warnings = state.get("enrichment_warnings", [])

    parts: list[str] = []

    parts.append(f"## Task\n{_format_task(task_raw)}")

    if ranked:
        parts.append(f"## Context ({len(ranked)} items)\n{_format_context(ranked)}")

    context_notes: list[str] = []
    if not has_code:
        context_notes.append("No code context available — plan based on task description only")
    if not has_doc:
        context_notes.append("No documentation context available")
    if has_similar:
        context_notes.append("Similar historical tasks found — review patterns in context")
    if warnings:
        context_notes.append(f"Warnings: {'; '.join(warnings)}")

    if context_notes:
        parts.append("## Notes\n" + "\n".join(f"- {n}" for n in context_notes))

    parts.append("## Instructions\nProduce a structured execution plan.")
    return "\n\n".join(parts)


def _format_task(task_raw: dict) -> str:
    title = task_raw.get("title", task_raw.get("free_text", "No title"))
    desc = task_raw.get("description", "")
    comments = task_raw.get("comments", [])
    lines = [f"**{title}**"]
    if desc:
        lines.append(desc)
    if comments:
        lines.append(f"Comments: {len(comments)} attached")
    return "\n".join(lines)


def _format_context(items: list[dict]) -> str:
    parts = []
    for i, item in enumerate(items[:10], 1):
        source = item.get("source", "unknown")
        content = str(item.get("content", ""))[:500]
        parts.append(f"[{i}] ({source}) {content}")
    return "\n".join(parts)
