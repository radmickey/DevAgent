"""Longterm memory: patterns + prompt versioning with SHA256 hash + timestamp."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_patterns_domain ON patterns(domain);
            CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
            CREATE INDEX IF NOT EXISTS idx_prompt_versions_node ON prompt_versions(node, active);
        """)
        self._conn.commit()

    # --- Prompt versioning ---

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

    def get_prompt_history(self, node: str) -> list[dict[str, Any]]:
        """Get full prompt history for a node."""
        cursor = self._conn.execute(
            "SELECT prompt_hash, reason, created_at, active FROM prompt_versions "
            "WHERE node = ? ORDER BY id DESC",
            (node,),
        )
        return [
            {"hash": row[0], "reason": row[1], "created_at": row[2], "active": bool(row[3])}
            for row in cursor.fetchall()
        ]

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

    # --- Pattern storage ---

    def save_pattern(
        self,
        domain: str,
        pattern_type: str,
        content: str,
        source_task: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Save a pattern learned from task history. Returns the pattern ID."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "INSERT INTO patterns (domain, pattern_type, content, source_task, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (domain, pattern_type, content, source_task, json.dumps(metadata or {}), now),
        )
        self._conn.commit()
        pattern_id = cursor.lastrowid or 0
        log.info("pattern_saved", domain=domain, type=pattern_type, id=pattern_id)
        return pattern_id

    def get_patterns(
        self, domain: str | None = None, pattern_type: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get patterns filtered by domain and/or type."""
        query = "SELECT id, domain, pattern_type, content, source_task, metadata, created_at FROM patterns"
        conditions: list[str] = []
        params: list[str] = []

        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if pattern_type:
            conditions.append("pattern_type = ?")
            params.append(pattern_type)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += f" ORDER BY id DESC LIMIT {limit}"

        cursor = self._conn.execute(query, params)
        return [
            {
                "id": row[0], "domain": row[1], "pattern_type": row[2],
                "content": row[3], "source_task": row[4],
                "metadata": json.loads(row[5]), "created_at": row[6],
            }
            for row in cursor.fetchall()
        ]

    def count_patterns(self) -> dict[str, int]:
        """Count patterns by domain."""
        cursor = self._conn.execute(
            "SELECT domain, COUNT(*) FROM patterns GROUP BY domain ORDER BY COUNT(*) DESC"
        )
        return {row[0]: row[1] for row in cursor.fetchall()}

    def close(self) -> None:
        self._conn.close()
