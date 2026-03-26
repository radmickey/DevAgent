"""Meta-Agent: Self-Evolution L1 — analyzes task outcomes and updates prompts.

Strategy:
1. After each task completion, Meta-Agent reviews:
   - Was the plan approved on first try or required HITL iterations?
   - Did the reviewer approve on first pass?
   - Were there recurring review findings?
2. If patterns indicate prompt weakness, it generates an improved prompt
   and saves it with full versioning (SHA256 hash + timestamp).
3. All prompt updates are reversible via rollback.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import BaseModel, Field

from agent.memory.longterm import LongtermMemory

log = structlog.get_logger()


class PromptUpdate(BaseModel):
    """A proposed prompt update from the Meta-Agent."""

    node: str
    old_hash: str | None = None
    new_prompt: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


CONFIDENCE_THRESHOLD = 0.7


async def maybe_update_prompts(
    task_id: str,
    feedback: str,
    task_outcome: dict[str, Any] | None = None,
    longterm: LongtermMemory | None = None,
) -> list[PromptUpdate]:
    """Analyze task feedback and outcome, propose prompt improvements.

    Only applies updates above the confidence threshold.
    Returns list of applied updates (may be empty).
    """
    if not feedback and not task_outcome:
        log.debug("meta_agent_skip", task_id=task_id, reason="no feedback or outcome")
        return []

    updates = _analyze_outcome(task_id, feedback, task_outcome or {})

    if not updates:
        log.info("meta_agent_no_updates", task_id=task_id)
        return []

    applied: list[PromptUpdate] = []
    for update in updates:
        if update.confidence < CONFIDENCE_THRESHOLD:
            log.info(
                "meta_agent_low_confidence",
                task_id=task_id,
                node=update.node,
                confidence=update.confidence,
            )
            continue

        if longterm:
            old_prompt = longterm.get_active_prompt(update.node)
            if old_prompt and old_prompt == update.new_prompt:
                continue

            update.old_hash = _get_current_hash(longterm, update.node)
            longterm.save_prompt_version(update.node, update.new_prompt, update.reason)
            longterm.save_pattern(
                domain="meta_agent",
                pattern_type="prompt_update",
                content=update.reason,
                source_task=task_id,
                metadata={"node": update.node, "confidence": update.confidence},
            )

        applied.append(update)
        log.info(
            "meta_agent_prompt_updated",
            task_id=task_id,
            node=update.node,
            confidence=update.confidence,
            reason=update.reason,
        )

    return applied


def _analyze_outcome(
    task_id: str, feedback: str, outcome: dict[str, Any]
) -> list[PromptUpdate]:
    """Analyze outcome and feedback to produce prompt update candidates.

    This is a rule-based analysis (L1). Future iterations will use LLM.
    """
    updates: list[PromptUpdate] = []

    review = outcome.get("review_result", {})
    iterations = outcome.get("iteration_count", 0)
    plan_approved = outcome.get("plan_approved", True)
    findings = review.get("findings", [])

    if iterations > 2:
        updates.append(PromptUpdate(
            node="executor",
            new_prompt=_add_instruction(
                "executor",
                "Pay extra attention to edge cases and error handling — "
                "previous executions required multiple review cycles.",
            ),
            reason=f"Task {task_id}: {iterations} review iterations needed",
            confidence=min(0.5 + iterations * 0.1, 0.95),
        ))

    if not plan_approved and feedback:
        updates.append(PromptUpdate(
            node="explainer",
            new_prompt=_add_instruction(
                "explainer",
                f"User feedback on rejected plan: {feedback[:200]}. "
                "Incorporate this pattern in future plans.",
            ),
            reason=f"Plan rejected with feedback: {feedback[:100]}",
            confidence=0.6,
        ))

    error_findings = [f for f in findings if f.get("severity") == "error"]
    if len(error_findings) >= 3:
        error_types = [f.get("message", "")[:50] for f in error_findings[:5]]
        updates.append(PromptUpdate(
            node="executor",
            new_prompt=_add_instruction(
                "executor",
                f"Common errors to avoid: {'; '.join(error_types)}",
            ),
            reason=f"Multiple error findings ({len(error_findings)})",
            confidence=0.75,
        ))

    if review.get("tests_passed") is False:
        updates.append(PromptUpdate(
            node="executor",
            new_prompt=_add_instruction(
                "executor",
                "Always ensure test compatibility. Run related tests mentally before writing code.",
            ),
            reason="Tests failed during review",
            confidence=0.8,
        ))

    return updates


def _add_instruction(node: str, instruction: str) -> str:
    """Create an augmented prompt by adding an instruction to the base prompt."""
    from agent.pipeline.prompts.explainer import EXPLAINER_SYSTEM_PROMPT
    from agent.pipeline.prompts.executor import EXECUTOR_SYSTEM_PROMPT
    from agent.pipeline.prompts.reviewer import REVIEWER_SYSTEM_PROMPT

    base_prompts = {
        "explainer": EXPLAINER_SYSTEM_PROMPT,
        "executor": EXECUTOR_SYSTEM_PROMPT,
        "reviewer": REVIEWER_SYSTEM_PROMPT,
    }
    base = base_prompts.get(node, "")
    return f"{base}\n\nAdditional instruction: {instruction}"


def _get_current_hash(longterm: LongtermMemory, node: str) -> str | None:
    """Get the current prompt hash for a node."""
    history = longterm.get_prompt_history(node)
    for entry in history:
        if entry.get("active"):
            return entry.get("hash")
    return None
