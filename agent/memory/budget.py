"""Token Budget Manager: tiktoken-based exact counting + fit_context()."""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger()

_enc = None

BUDGET: dict[str, int] = {
    "system_prompt": 2_000,
    "tools": 3_000,
    "task": 2_000,
    "context": 8_000,
    "docs": 1_000,
}


def _get_encoder():
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    return len(_get_encoder().encode(text))


def fit_context(items: list[Any], max_tokens: int = BUDGET["context"]) -> list[Any]:
    """Select items that fit within the token budget, sorted by relevance.

    Each item must have .content (str) and .relevance (float) attributes,
    or be a dict with "content" and "relevance" keys.
    """
    def _get_content(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("content", str(item)))
        return str(getattr(item, "content", str(item)))

    def _get_relevance(item: Any) -> float:
        if isinstance(item, dict):
            return float(item.get("relevance", 0.0))
        return float(getattr(item, "relevance", 0.0))

    sorted_items = sorted(items, key=_get_relevance, reverse=True)

    result: list[Any] = []
    used = 0
    for item in sorted_items:
        tokens = count_tokens(_get_content(item))
        if used + tokens > max_tokens:
            log.warning(
                "budget_overflow",
                max_tokens=max_tokens,
                used=used,
                skipped_tokens=tokens,
                remaining_items=len(sorted_items) - len(result),
            )
            break
        result.append(item)
        used += tokens

    log.info("budget_fit", items_kept=len(result), tokens_used=used, max_tokens=max_tokens)
    return result
