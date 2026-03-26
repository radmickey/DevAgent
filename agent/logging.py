"""structlog configuration: dev (pretty) + prod (JSON) formatters."""

from typing import Any

import structlog

from agent.config import get_config


def setup_logging() -> None:
    """Configure structlog: pretty for dev, JSON for prod."""
    config = get_config()

    renderer: Any
    if config.env == "prod":
        renderer = structlog.JSONRenderer()  # type: ignore[operator]
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
