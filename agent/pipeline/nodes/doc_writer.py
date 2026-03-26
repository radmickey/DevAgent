"""DocWriter node: writes documentation as JSON Lines + stores in ChromaDB.

After task completion:
1. Writes a JSON Lines entry with task summary, plan, changes, review
2. Stores the documentation in ChromaDB for future context retrieval
3. Supports lang filter via metadata for language-specific search
"""

from __future__ import annotations

from typing import Any

import structlog

from agent.memory.agent_docs import AgentDocs
from agent.pipeline.state import PipelineState

log = structlog.get_logger()


async def doc_writer_node(
    state: PipelineState,
    *,
    docs: AgentDocs | None = None,
    vector_memory: Any = None,
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

    summary = _build_summary(task_raw, plan, code_changes, review)
    details = _build_details(plan, code_changes, review)
    languages = _detect_languages(code_changes)

    if docs is None:
        docs = AgentDocs()

    try:
        docs.write(task_id, summary, details)
        log.info("doc_written", task_id=task_id, languages=languages)
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
