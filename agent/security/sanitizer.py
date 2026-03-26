"""Sanitizer: clean sensitive data before sending to external LLMs."""

from __future__ import annotations

import re

import structlog

log = structlog.get_logger()

_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)(api[_-]?key|secret|token|password|credential)\s*[=:]\s*\S+", r"\1=***REDACTED***"),
    (r"(?i)bearer\s+\S+", "Bearer ***REDACTED***"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "***EMAIL***"),
]


def sanitize(text: str) -> str:
    """Remove potentially sensitive data from text before sending to external LLM."""
    result = text
    for pattern, replacement in _PATTERNS:
        result = re.sub(pattern, replacement, result)
    return result


def should_sanitize() -> bool:
    """Check if external LLM is being used (requires sanitization)."""
    from agent.config import get_config
    return get_config().use_external_llm
