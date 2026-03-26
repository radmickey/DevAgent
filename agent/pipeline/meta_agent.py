"""Meta-Agent: Self-Evolution L1+L2 — analyzes task outcomes and updates prompts.

Strategy:
L1 (rule-based, always available):
  1. After each task completion, Meta-Agent reviews:
     - Was the plan approved on first try or required HITL iterations?
     - Did the reviewer approve on first pass?
     - Were there recurring review findings?
  2. If patterns indicate prompt weakness, it proposes an improved prompt.

L2 (LLM-based, optional):
  Uses the meta_agent system prompt to analyze outcomes and propose
  targeted prompt improvements with reasoning.

Both levels:
  3. All prompt updates are saved with full versioning (SHA256 hash + timestamp).
  4. All prompt updates are reversible via rollback.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agent.llm import get_model_for_node
from agent.memory.longterm import LongtermMemory
from agent.pipeline.prompts import PROMPTS, get_prompt

log = structlog.get_logger()

META_AGENT_SYSTEM_PROMPT = get_prompt("meta_agent")


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
    use_llm: bool = False,
) -> list[PromptUpdate]:
    """Analyze task feedback and outcome, propose prompt improvements.

    L1 (rule-based) runs always. L2 (LLM) runs when use_llm=True.
    Only applies updates above the confidence threshold.
    Returns list of applied updates (may be empty).
    """
    if not feedback and not task_outcome:
        log.debug("meta_agent_skip", task_id=task_id, reason="no feedback or outcome")
        return []

    updates = _analyze_outcome(task_id, feedback, task_outcome or {})

    if use_llm:
        llm_updates = await _analyze_outcome_llm(task_id, feedback, task_outcome or {})
        updates.extend(llm_updates)

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


async def _analyze_outcome_llm(
    task_id: str, feedback: str, outcome: dict[str, Any],
) -> list[PromptUpdate]:
    """L2: Use LLM with meta_agent prompt to propose prompt updates."""
    nodes_to_evaluate = ["explainer", "executor", "reviewer"]
    updates: list[PromptUpdate] = []

    for node_name in nodes_to_evaluate:
        current_prompt = PROMPTS.get(node_name, "")
        if not current_prompt:
            continue

        user_prompt = _build_meta_agent_prompt(node_name, current_prompt, outcome, feedback)
        try:
            agent: Agent[None, str] = Agent(
                "test",
                output_type=str,
                system_prompt=META_AGENT_SYSTEM_PROMPT,
            )
            model = get_model_for_node("meta_agent")
            result = await agent.run(user_prompt, model=model)
            parsed = json.loads(result.output)

            if parsed.get("should_update") and parsed.get("suggested_prompt"):
                updates.append(PromptUpdate(
                    node=node_name,
                    new_prompt=parsed["suggested_prompt"],
                    reason=parsed.get("reason", "LLM-suggested improvement"),
                    confidence=0.75,
                ))
                log.info(
                    "meta_agent_llm_suggestion",
                    node=node_name,
                    reason=parsed.get("reason", ""),
                    changes=parsed.get("changes_summary", ""),
                )
        except Exception as exc:
            log.warning("meta_agent_llm_failed", node=node_name, error=str(exc))

    return updates


def _build_meta_agent_prompt(
    node_name: str,
    current_prompt: str,
    outcome: dict[str, Any],
    feedback: str,
) -> str:
    """Build user prompt for meta-agent LLM evaluation."""
    return (
        f"## Node: {node_name}\n\n"
        f"## Current prompt\n```\n{current_prompt[:3000]}\n```\n\n"
        f"## Recent outcome\n```json\n{json.dumps(outcome, default=str)[:2000]}\n```\n\n"
        f"## User feedback\n{feedback[:500] if feedback else 'No feedback.'}\n\n"
        "Evaluate whether the prompt needs updating based on this outcome."
    )


def _analyze_outcome(
    task_id: str, feedback: str, outcome: dict[str, Any]
) -> list[PromptUpdate]:
    """L1: rule-based analysis of outcome to produce prompt update candidates."""
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
    base = PROMPTS.get(node, "")
    return f"{base}\n\nAdditional instruction: {instruction}"


def _get_current_hash(longterm: LongtermMemory, node: str) -> str | None:
    """Get the current prompt hash for a node."""
    history = longterm.get_prompt_history(node)
    for entry in history:
        if entry.get("active"):
            return entry.get("hash")
    return None
