"""Session memory: SqliteSaver wrapper for LangGraph checkpointing."""

from __future__ import annotations


import structlog

from agent.config import DEVAGENT_HOME

log = structlog.get_logger()

_SESSION_DB_PATH = DEVAGENT_HOME / "sessions.db"


def get_session_db_path() -> str:
    """Return the path to the session SQLite database."""
    return str(_SESSION_DB_PATH)
