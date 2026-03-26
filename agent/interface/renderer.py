"""Rich renderers: progress bars, plan display, warnings."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def render_plan(plan: dict) -> None:
    """Render an execution plan as a Rich panel."""
    table = Table(title="Execution Plan", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Action", style="cyan")
    table.add_column("File", style="green")
    table.add_column("Description")

    steps = plan.get("steps", [])
    for i, step in enumerate(steps, 1):
        table.add_row(
            str(i),
            step.get("action", ""),
            step.get("file_path", ""),
            step.get("description", ""),
        )

    console.print(Panel(table, title=plan.get("summary", "Plan")))


def render_warnings(warnings: list[str]) -> None:
    """Render enrichment warnings."""
    if not warnings:
        return
    for w in warnings:
        console.print(f"  [yellow]⚠ {w}[/yellow]")


def render_review(review: dict) -> None:
    """Render review results."""
    approved = review.get("approved", False)
    status = "[green]✓ APPROVED[/green]" if approved else "[red]✗ REJECTED[/red]"
    console.print(f"\n[bold]Review:[/bold] {status}")
    for finding in review.get("findings", []):
        severity = finding.get("severity", "info")
        color = {"error": "red", "warning": "yellow"}.get(severity, "dim")
        console.print(f"  [{color}]{severity}: {finding.get('message', '')}[/{color}]")
