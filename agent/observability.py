"""Observability: LangSmith + Pydantic Logfire tracing setup.

Enables full tracing of every LLM call when environment variables are configured:
- LANGSMITH_API_KEY → LangSmith tracing
- LOGFIRE_TOKEN → Pydantic Logfire tracing

Both are optional — agent works without any tracing configured.
"""

from __future__ import annotations

import os

import structlog

log = structlog.get_logger()

_langsmith_initialized = False
_logfire_initialized = False


def setup_tracing() -> dict[str, bool]:
    """Initialize tracing backends. Returns dict of which backends are active."""
    active: dict[str, bool] = {"langsmith": False, "logfire": False}

    active["langsmith"] = _setup_langsmith()
    active["logfire"] = _setup_logfire()

    if not any(active.values()):
        log.info("observability_disabled", hint="Set LANGSMITH_API_KEY or LOGFIRE_TOKEN to enable")
    else:
        log.info("observability_enabled", backends=active)

    return active


def _setup_langsmith() -> bool:
    """Initialize LangSmith tracing if API key is set."""
    global _langsmith_initialized
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return False

    if _langsmith_initialized:
        return True

    try:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", os.getenv("LANGSMITH_PROJECT", "devagent"))

        _langsmith_initialized = True
        log.info(
            "langsmith_enabled",
            project=os.environ.get("LANGSMITH_PROJECT"),
        )
        return True
    except Exception as exc:
        log.warning("langsmith_init_failed", error=str(exc))
        return False


def _setup_logfire() -> bool:
    """Initialize Pydantic Logfire tracing if token is set."""
    global _logfire_initialized
    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        return False

    if _logfire_initialized:
        return True

    try:
        import logfire
        logfire.configure(token=token)
        logfire.instrument_pydantic_ai()

        _logfire_initialized = True
        log.info("logfire_enabled")
        return True
    except ImportError:
        log.warning("logfire_not_installed", hint="pip install logfire[pydantic-ai]")
        return False
    except Exception as exc:
        log.warning("logfire_init_failed", error=str(exc))
        return False


def get_tracing_status() -> dict[str, bool]:
    """Get current tracing status without initializing."""
    return {
        "langsmith": _langsmith_initialized,
        "logfire": _logfire_initialized,
    }
