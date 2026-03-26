"""Agent documentation: JSON Lines storage + ChromaDB search."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import structlog

from agent.config import DEVAGENT_HOME

log = structlog.get_logger()

_DOCS_PATH = DEVAGENT_HOME / "agent_docs.jsonl"


class AgentDocs:
    """JSON Lines documentation writer for completed tasks."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DOCS_PATH

    def write(self, task_id: str, summary: str, details: dict) -> None:
        """Append a documentation entry as JSON Lines."""
        entry = {
            "task_id": task_id,
            "summary": summary,
            "details": details,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.info("doc_written", task_id=task_id)

    def read_all(self) -> list[dict]:
        """Read all documentation entries."""
        if not self._path.exists():
            return []
        entries = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
