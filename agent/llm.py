"""Dual LLM configuration: fast for enricher, strong for explainer/executor/reviewer.

Fast model: ~60% cheaper, used for routine tasks (enrichment, sub-agents)
Strong model: used for planning, code generation, and review
"""

from __future__ import annotations

import structlog

from agent.config import get_config

log = structlog.get_logger()

_NODE_MODEL_MAP: dict[str, str] = {
    "input_router": "fast",
    "reader": "fast",
    "enricher": "fast",
    "ranker": "fast",
    "code_search": "fast",
    "doc_search": "fast",
    "task_search": "fast",
    "diff_agent": "fast",
    "explainer": "strong",
    "executor": "strong",
    "reviewer": "strong",
    "doc_writer": "fast",
    "pattern_extractor": "strong",
    "meta_agent": "strong",
}


def get_model_for_node(node: str) -> str:
    """Return the model identifier for a given node.

    Returns the full model name (e.g. 'claude-sonnet-4-20250514')
    based on whether the node needs fast or strong LLM.
    """
    config = get_config()
    tier = _NODE_MODEL_MAP.get(node, "fast")

    if tier == "strong":
        model = config.llm_strong_model
    else:
        model = config.llm_fast_model

    log.debug("llm_model_selected", node=node, tier=tier, model=model)
    return model


def get_fast_model() -> str:
    """Get the fast (cheap) model name."""
    return get_config().llm_fast_model


def get_strong_model() -> str:
    """Get the strong (capable) model name."""
    return get_config().llm_strong_model
