"""Side effect tracker: records all external side effects for rollback."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import structlog

from agent.config import DEVAGENT_HOME

log = structlog.get_logger()

_DB_PATH = DEVAGENT_HOME / "effects.db"

ROLLBACK_ACTIONS: dict[str, str] = {
    "branch_created": "delete_branch",
    "file_written": "delete_file",
    "commit_pushed": "revert_commit",
    "status_updated": "revert_status",
    "comment_added": "delete_comment",
}


class SideEffectTracker:
    """Track side effects per task for auditability and rollback."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DB_PATH
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS side_effects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                effect_type TEXT NOT NULL,
                details TEXT NOT NULL,
                created_at TEXT NOT NULL,
                interrupted INTEGER DEFAULT 0
            )
        """)
        self._conn.commit()

    def record(self, task_id: str, effect_type: str, details: dict) -> None:
        """Record a side effect."""
        self._conn.execute(
            "INSERT INTO side_effects (task_id, effect_type, details, created_at) VALUES (?, ?, ?, ?)",
            (task_id, effect_type, json.dumps(details), datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()
        log.info("effect_recorded", task_id=task_id, effect_type=effect_type)

    def get(self, task_id: str) -> list[dict]:
        """Get all side effects for a task."""
        cursor = self._conn.execute(
            "SELECT effect_type, details, created_at, interrupted FROM side_effects WHERE task_id = ? ORDER BY id",
            (task_id,),
        )
        return [
            {
                "effect_type": row[0],
                "details": json.loads(row[1]),
                "created_at": row[2],
                "interrupted": bool(row[3]),
            }
            for row in cursor.fetchall()
        ]

    def get_rollback_plan(self, task_id: str) -> list[dict]:
        """Generate a rollback plan for a task's side effects (reverse order)."""
        effects = self.get(task_id)
        plan = []
        for effect in reversed(effects):
            rollback_action = ROLLBACK_ACTIONS.get(effect["effect_type"])
            if rollback_action:
                plan.append({
                    "action": rollback_action,
                    "original_effect": effect["effect_type"],
                    "details": effect["details"],
                })
        return plan

    def mark_interrupted(self, task_id: str | None = None) -> None:
        """Mark all unfinished effects as interrupted (for graceful shutdown)."""
        if task_id:
            self._conn.execute(
                "UPDATE side_effects SET interrupted = 1 WHERE task_id = ? AND interrupted = 0",
                (task_id,),
            )
        else:
            self._conn.execute(
                "UPDATE side_effects SET interrupted = 1 WHERE interrupted = 0"
            )
        self._conn.commit()
        log.warning("effects_marked_interrupted", task_id=task_id)

    def close(self) -> None:
        self._conn.close()
