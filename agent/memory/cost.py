"""Cost Tracker: SQLite log of every LLM call with model/tokens/latency/cost."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import structlog

from agent.config import DEVAGENT_HOME

log = structlog.get_logger()

_DB_PATH = DEVAGENT_HOME / "cost.db"


class CostTracker:
    """Track LLM usage costs per task and node."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DB_PATH
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                node TEXT NOT NULL,
                model TEXT NOT NULL,
                tokens_in INTEGER NOT NULL,
                tokens_out INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def log(
        self,
        task_id: str,
        node: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: int,
        cost_usd: float,
    ) -> None:
        """Log a single LLM call."""
        self._conn.execute(
            "INSERT INTO llm_costs (task_id, node, model, tokens_in, tokens_out, latency_ms, cost_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, node, model, tokens_in, tokens_out, latency_ms, cost_usd,
             datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()
        log.debug(
            "cost_logged",
            task_id=task_id,
            node=node,
            model=model,
            tokens=tokens_in + tokens_out,
            cost_usd=cost_usd,
        )

    def get_task_cost(self, task_id: str) -> float:
        """Get total cost for a task."""
        cursor = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_costs WHERE task_id = ?",
            (task_id,),
        )
        row = cursor.fetchone()
        return float(row[0]) if row else 0.0

    def get_total_cost(self) -> float:
        """Get total cost across all tasks."""
        cursor = self._conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM llm_costs")
        row = cursor.fetchone()
        return float(row[0]) if row else 0.0

    def close(self) -> None:
        self._conn.close()
