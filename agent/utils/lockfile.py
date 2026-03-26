"""PID lockfile — one DevAgent process at a time."""

from __future__ import annotations

import os

from agent.config import DEVAGENT_HOME


class AgentLock:
    """Context manager that enforces single-process execution via PID lockfile."""

    LOCK_PATH = DEVAGENT_HOME / "agent.lock"

    def _is_process_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def __enter__(self) -> "AgentLock":
        if self.LOCK_PATH.exists():
            raw = self.LOCK_PATH.read_text().strip()
            try:
                pid = int(raw)
            except ValueError:
                self.LOCK_PATH.unlink()
            else:
                if self._is_process_alive(pid):
                    raise RuntimeError(
                        f"DevAgent already running (PID {pid}).\n"
                        "Wait for it to finish or remove ~/.devagent/agent.lock"
                    )
                self.LOCK_PATH.unlink()

        self.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.LOCK_PATH.write_text(str(os.getpid()))
        return self

    def __exit__(self, *_: object) -> None:
        self.LOCK_PATH.unlink(missing_ok=True)
