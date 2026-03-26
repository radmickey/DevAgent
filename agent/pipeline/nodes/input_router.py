"""Input Router node: determines input type (task_id / text / both) and normalizes.

Uses rule-based detection by default; LLM classifier (via classifier prompt)
as an optional enrichment for task_type, language_hints, and urgency.
"""

from __future__ import annotations

import json
import re

import structlog
from pydantic_ai import Agent

from agent.llm import get_model_for_node
from agent.pipeline.prompts import get_prompt
from agent.pipeline.state import PipelineState

log = structlog.get_logger()

CLASSIFIER_SYSTEM_PROMPT = get_prompt("classifier")


def _looks_like_task_id(text: str) -> bool:
    """Check if input looks like a tracker task ID (e.g. PROJ-123, #456)."""
    return bool(re.match(r"^[A-Z]+-\d+$", text.strip())) or bool(
        re.match(r"^#?\d+$", text.strip())
    )


async def input_router_node(
    state: PipelineState,
    *,
    use_llm_classifier: bool = False,
) -> PipelineState:
    """Parse raw input, detect type, populate task_id in state.

    When use_llm_classifier is True, runs the classifier prompt for
    richer metadata (task_type, language_hints, urgency).
    """
    task_id = state.get("task_id", "")

    if not task_id:
        log.error("input_router_no_input")
        return {**state, "task_id": "", "task_raw": {"error": "no input provided"}}

    raw_input = task_id.strip()
    resolved_id: str
    task_raw: dict

    if _looks_like_task_id(raw_input):
        log.info("input_router_task_id", task_id=raw_input)
        resolved_id = raw_input
        task_raw = {}
    else:
        log.info("input_router_free_text", text=raw_input[:80])
        resolved_id = ""
        task_raw = {"free_text": raw_input}

    if use_llm_classifier:
        classification = await _classify_with_llm(raw_input)
        if classification:
            task_raw.update({
                k: v for k, v in classification.items()
                if k in ("task_type", "language_hints", "urgency", "confidence")
                and v is not None
            })
            if classification.get("task_id") and not resolved_id:
                resolved_id = classification["task_id"]
            if classification.get("task_text") and "free_text" not in task_raw:
                task_raw["free_text"] = classification["task_text"]

    return {
        **state,
        "task_id": resolved_id,
        "task_raw": task_raw,
    }


async def _classify_with_llm(raw_input: str) -> dict | None:
    """Run the classifier LLM to enrich input metadata."""
    try:
        agent: Agent[None, str] = Agent(
            "test",
            output_type=str,
            system_prompt=CLASSIFIER_SYSTEM_PROMPT,
        )
        model = get_model_for_node("input_router")
        result = await agent.run(raw_input, model=model)
        parsed: dict = json.loads(result.output)
        return parsed
    except Exception as exc:
        log.warning("input_router_llm_classify_failed", error=str(exc))
        return None
