"""Sanitizer: clean sensitive data before sending to external LLMs.

When USE_EXTERNAL_LLM=true, all prompts are sanitized before sending.
"""

from __future__ import annotations

import re

import structlog

log = structlog.get_logger()

_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)(api[_-]?key|secret[_-]?key|token|password|credential|auth[_-]?token)\s*[=:]\s*\S+",
     r"\1=***REDACTED***"),
    (r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer ***REDACTED***"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "***EMAIL***"),
    (r"(?i)(aws_access_key_id|aws_secret_access_key)\s*=\s*\S+", r"\1=***REDACTED***"),
    (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "***IP***"),
    (r"(?i)(jdbc|postgresql|mysql|mongodb)://\S+", "***DB_URL***"),
    (r"-----BEGIN\s+(RSA\s+)?PRIVATE KEY-----[\s\S]*?-----END\s+(RSA\s+)?PRIVATE KEY-----",
     "***PRIVATE_KEY***"),
]


def sanitize(text: str) -> str:
    """Remove potentially sensitive data from text before sending to external LLM."""
    result = text
    for pattern, replacement in _PATTERNS:
        result = re.sub(pattern, replacement, result)
    return result


def sanitize_if_needed(text: str) -> str:
    """Sanitize only if USE_EXTERNAL_LLM is true."""
    if should_sanitize():
        return sanitize(text)
    return text


def should_sanitize() -> bool:
    """Check if external LLM is being used (requires sanitization)."""
    from agent.config import get_config
    return get_config().use_external_llm
