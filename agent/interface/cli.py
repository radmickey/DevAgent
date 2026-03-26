"""CLI interface: Typer + Rich for DevAgent commands."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(name="devagent", help="DevAgent — AI-powered developer task automation")
console = Console()


@app.command()
def run(
    task_id: str = typer.Argument(..., help="Task ID (e.g. PROJ-123) or free text"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate plan but don't execute"),
) -> None:
    """Run the DevAgent pipeline on a task."""
    from agent.logging import setup_logging
    from agent.main import run as main_run

    setup_logging()
    console.print(f"[bold]DevAgent[/bold] processing: {task_id}")
    if dry_run:
        console.print("[yellow]--dry-run mode: will stop before executor[/yellow]")
    main_run()


@app.command()
def rollback(
    task_id: str = typer.Argument(..., help="Task ID to rollback"),
) -> None:
    """Rollback side effects for a task."""
    from agent.memory.effects import SideEffectTracker

    tracker = SideEffectTracker()
    plan = tracker.get_rollback_plan(task_id)
    if not plan:
        console.print(f"[green]No side effects to rollback for {task_id}[/green]")
        return

    console.print(f"[yellow]Rollback plan for {task_id}:[/yellow]")
    for step in plan:
        console.print(f"  - {step['action']}: {step['details']}")
    tracker.close()


@app.command()
def status() -> None:
    """Show DevAgent status and cost summary."""
    from agent.memory.cost import CostTracker

    ct = CostTracker()
    total = ct.get_total_cost()
    console.print(f"[bold]Total LLM cost:[/bold] ${total:.4f}")
    ct.close()


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
) -> None:
    """Start the web dashboard (FastAPI + SSE)."""
    try:
        import uvicorn
        from agent.interface.web import create_app
    except ImportError:
        console.print("[red]Web dependencies not installed. Run: pip install 'devagent[web]'[/red]")
        raise typer.Exit(1)

    web_app = create_app()
    console.print(f"[bold]DevAgent Web[/bold] → http://{host}:{port}")
    uvicorn.run(web_app, host=host, port=port)


def main() -> None:
    app()
