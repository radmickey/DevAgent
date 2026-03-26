"""DocWriter node: writes documentation as JSON Lines + stores in ChromaDB.

After task completion:
1. Optionally generates structured doc entry via LLM (using doc_writer prompt)
2. Writes a JSON Lines entry with task summary, plan, changes, review
3. Stores the documentation in ChromaDB for future context retrieval
4. Supports lang filter via metadata for language-specific search
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from pydantic_ai import Agent

from agent.llm import get_model_for_node
from agent.memory.agent_docs import AgentDocs
from agent.pipeline.prompts import get_prompt
from agent.pipeline.state import PipelineState

log = structlog.get_logger()

DOC_WRITER_SYSTEM_PROMPT = get_prompt("doc_writer")


async def doc_writer_node(
    state: PipelineState,
    *,
    docs: AgentDocs | None = None,
    vector_memory: Any = None,
    use_llm: bool = False,
) -> PipelineState:
    """Write documentation for the completed task."""
    task_id = state.get("task_id", "")
    plan: dict = state.get("plan", {})
    code_changes = state.get("code_changes", [])
    review = state.get("review_result", {})
    task_raw = state.get("task_raw", {})

    if not task_id:
        log.warning("doc_writer_no_task_id")
        return state

    llm_doc: dict[str, Any] | None = None
    if use_llm:
        llm_doc = await _generate_llm_doc(task_raw, plan, code_changes, review)

    summary = _build_summary(task_raw, plan, code_changes, review)
    details = _build_details(plan, code_changes, review)
    if llm_doc:
        details["llm_doc_entry"] = llm_doc
    languages = _detect_languages(code_changes)

    if docs is None:
        docs = AgentDocs()

    try:
        docs.write(task_id, summary, details)
        log.info("doc_written", task_id=task_id, languages=languages, llm_doc=llm_doc is not None)
    except Exception as exc:
        log.error("doc_write_failed", task_id=task_id, error=str(exc))

    if vector_memory is not None:
        try:
            items = _build_vector_items(task_id, summary, details, languages)
            vector_memory.store(task_id, items)
            log.info("doc_vectorized", task_id=task_id, items=len(items))
        except Exception as exc:
            log.error("doc_vectorize_failed", task_id=task_id, error=str(exc))

    return state


async def _generate_llm_doc(
    task_raw: dict,
    plan: dict,
    code_changes: list,
    review: dict,
) -> dict[str, Any] | None:
    """Generate a structured doc entry via LLM using the doc_writer prompt."""
    user_prompt = _build_llm_doc_prompt(task_raw, plan, code_changes, review)
    try:
        agent: Agent[None, str] = Agent(
            "test",
            output_type=str,
            system_prompt=DOC_WRITER_SYSTEM_PROMPT,
        )
        model = get_model_for_node("doc_writer")
        result = await agent.run(user_prompt, model=model)
        parsed: dict[str, Any] = json.loads(result.output)
        return parsed
    except Exception as exc:
        log.warning("doc_writer_llm_failed", error=str(exc))
        return None


def _build_llm_doc_prompt(
    task_raw: dict, plan: dict, code_changes: list, review: dict,
) -> str:
    """Build user prompt for the doc_writer LLM call."""
    title = task_raw.get("title", task_raw.get("free_text", "Unknown task"))
    parts = [f"## Task\n{title}"]
    if plan:
        parts.append(f"## Plan\nSummary: {plan.get('summary', '')}\n"
                      f"Steps: {len(plan.get('steps', []))}")
    if code_changes:
        files = [c.get("path", "?") for c in code_changes if isinstance(c, dict)]
        parts.append(f"## Files changed\n{', '.join(files)}")
    if review:
        parts.append(f"## Review\nApproved: {review.get('approved', '?')}\n"
                      f"Findings: {len(review.get('findings', []))}")
    parts.append("\nGenerate the documentation entry as described in your system prompt.")
    return "\n\n".join(parts)


def _build_summary(
    task_raw: dict, plan: dict, code_changes: list, review: dict
) -> str:
    """Build a human-readable summary of the task."""
    title = task_raw.get("title", task_raw.get("free_text", "Unknown task"))
    approach = plan.get("approach", "") if isinstance(plan, dict) else ""
    n_files = len(code_changes) if code_changes else 0
    approved = review.get("approved", None) if isinstance(review, dict) else None

    parts = [f"Task: {title}"]
    if approach:
        parts.append(f"Approach: {approach}")
    parts.append(f"Files changed: {n_files}")
    if approved is not None:
        parts.append(f"Review: {'approved' if approved else 'rejected'}")
    return ". ".join(parts)


def _build_details(plan: dict, code_changes: list, review: dict) -> dict:
    """Build detailed documentation structure."""
    return {
        "plan": plan if isinstance(plan, dict) else {},
        "files_changed": [
            {"path": c.get("path", ""), "action": c.get("action", "")}
            for c in (code_changes or [])
            if isinstance(c, dict)
        ],
        "review": {
            "approved": review.get("approved") if isinstance(review, dict) else None,
            "findings_count": len(review.get("findings", [])) if isinstance(review, dict) else 0,
        },
    }


def _detect_languages(code_changes: list) -> list[str]:
    """Detect programming languages from file extensions in changes."""
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".go": "go", ".rs": "rust", ".java": "java",
        ".rb": "ruby", ".cpp": "cpp", ".c": "c",
        ".kt": "kotlin", ".swift": "swift", ".cs": "csharp",
    }
    langs: set[str] = set()
    for change in code_changes or []:
        if not isinstance(change, dict):
            continue
        path = change.get("path", "")
        for ext, lang in ext_map.items():
            if path.endswith(ext):
                langs.add(lang)
                break
    return sorted(langs)


def _build_vector_items(
    task_id: str, summary: str, details: dict, languages: list[str]
) -> list[dict[str, Any]]:
    """Build items for ChromaDB storage with lang metadata."""
    items: list[dict[str, Any]] = [
        {
            "content": summary,
            "source": "doc_summary",
            "lang": ",".join(languages) if languages else "unknown",
        }
    ]

    plan = details.get("plan", {})
    if plan.get("steps"):
        steps_text = "\n".join(
            f"- {s.get('description', '')}" for s in plan["steps"] if isinstance(s, dict)
        )
        items.append({
            "content": f"Plan steps:\n{steps_text}",
            "source": "doc_plan",
            "lang": ",".join(languages) if languages else "unknown",
        })

    return items
