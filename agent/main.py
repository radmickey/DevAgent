"""DevAgent entry point: lockfile + signal handler + pipeline run."""

from __future__ import annotations

import signal
import sys

from rich.console import Console

from agent.memory.effects import SideEffectTracker
from agent.utils.lockfile import AgentLock

console = Console()


def setup_signal_handlers(lock: AgentLock, effects: SideEffectTracker) -> None:
    """Register SIGINT/SIGTERM handlers for graceful shutdown."""

    def _shutdown(sig: int, frame: object) -> None:
        console.print("\n[yellow]⚠ Interrupting...[/yellow]")
        effects.mark_interrupted()
        lock.__exit__(None, None, None)
        console.print("[dim]Lockfile released. For rollback: devagent rollback <task_id>[/dim]")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)


def run() -> None:
    """Main entry point."""
    from agent.logging import setup_logging

    setup_logging()

    lock = AgentLock()
    effects = SideEffectTracker()

    with lock:
        setup_signal_handlers(lock, effects)
        console.print("[green]DevAgent started[/green]")
        # Pipeline execution is wired in via CLI
