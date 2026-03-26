"""Longterm memory: patterns + prompt versioning with SHA256 hash + timestamp."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import structlog

from agent.config import DEVAGENT_HOME

log = structlog.get_logger()

_DB_PATH = DEVAGENT_HOME / "longterm.db"


class LongtermMemory:
    """SQLite-backed longterm memory for patterns and prompt versions."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DB_PATH
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS prompt_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                active INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                content TEXT NOT NULL,
                source_task TEXT,
                created_at TEXT NOT NULL
            );
        """)
        self._conn.commit()

    def save_prompt_version(self, node: str, prompt: str, reason: str) -> str:
        """Save a new prompt version, deactivate the previous one. Returns hash."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            "UPDATE prompt_versions SET active = 0 WHERE node = ? AND active = 1",
            (node,),
        )
        self._conn.execute(
            "INSERT INTO prompt_versions (node, prompt_hash, prompt_text, reason, created_at, active) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (node, prompt_hash, prompt, reason, now),
        )
        self._conn.commit()
        log.info("prompt_version_saved", node=node, hash=prompt_hash)
        return prompt_hash

    def get_active_prompt(self, node: str) -> str | None:
        """Get the currently active prompt for a node."""
        cursor = self._conn.execute(
            "SELECT prompt_text FROM prompt_versions WHERE node = ? AND active = 1",
            (node,),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def rollback_prompt(self, node: str, target_hash: str) -> bool:
        """Rollback to a specific prompt version by hash."""
        cursor = self._conn.execute(
            "SELECT id FROM prompt_versions WHERE node = ? AND prompt_hash = ?",
            (node, target_hash),
        )
        if not cursor.fetchone():
            return False

        self._conn.execute(
            "UPDATE prompt_versions SET active = 0 WHERE node = ?",
            (node,),
        )
        self._conn.execute(
            "UPDATE prompt_versions SET active = 1 WHERE node = ? AND prompt_hash = ?",
            (node, target_hash),
        )
        self._conn.commit()
        log.info("prompt_rollback", node=node, target_hash=target_hash)
        return True

    def close(self) -> None:
        self._conn.close()
