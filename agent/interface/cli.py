"""CLI interface: Typer + Rich for DevAgent commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="devagent",
    help="DevAgent — AI-powered developer task automation",
    invoke_without_command=True,
    no_args_is_help=True,
)
config_app = typer.Typer(name="config", help="Manage DevAgent settings", no_args_is_help=True)
app.add_typer(config_app, name="config")
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


# ---------------------------------------------------------------------------
# devagent config list / get / set
# ---------------------------------------------------------------------------

@config_app.command("list")
def config_list() -> None:
    """Show all settings with current values and sources."""
    from agent.config import get_all_settings

    settings = get_all_settings()

    table = Table(title="DevAgent Settings", show_lines=True)
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold")
    table.add_column("Source", style="dim")
    table.add_column("Env Var", style="dim")
    table.add_column("Default", style="dim")

    for name in sorted(settings):
        info = settings[name]
        value_str = str(info["value"])
        source = info["source"]

        if info["type"] == "bool":
            value_style = "[green]true[/green]" if info["value"] else "[red]false[/red]"
        else:
            value_style = value_str

        source_style = source
        if source == "settings.yaml":
            source_style = "[yellow]settings.yaml[/yellow]"
        elif source.startswith("env:"):
            source_style = f"[blue]{source}[/blue]"

        table.add_row(name, value_style, source_style, info["env_var"], str(info["default"]))

    console.print(table)


@config_app.command("get")
def config_get(
    name: str = typer.Argument(..., help="Setting name"),
) -> None:
    """Get the value of a specific setting."""
    from agent.config import get_all_settings

    settings = get_all_settings()
    if name not in settings:
        console.print(f"[red]Unknown setting: {name}[/red]")
        console.print(f"Available: {', '.join(sorted(settings.keys()))}")
        raise typer.Exit(1)

    info = settings[name]
    console.print(f"[cyan]{name}[/cyan] = [bold]{info['value']}[/bold]  (source: {info['source']})")


@config_app.command("set")
def config_set(
    name: str = typer.Argument(..., help="Setting name"),
    value: str = typer.Argument(..., help="New value (true/false for booleans)"),
) -> None:
    """Set a setting value. Persists to ~/.devagent/settings.yaml."""
    from agent.config import set_setting

    try:
        result = set_setting(name, value)
        console.print(
            f"[green]Updated[/green] [cyan]{result['name']}[/cyan] = "
            f"[bold]{result['value']}[/bold]  (saved to {result['source']})"
        )
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@config_app.command("reset")
def config_reset(
    name: str = typer.Argument(..., help="Setting name to reset to default"),
) -> None:
    """Remove a setting override, reverting to env/default."""
    from agent.config import _invalidate_config_cache, _load_settings_yaml, _save_settings_yaml

    yaml_settings = _load_settings_yaml()
    if name in yaml_settings:
        del yaml_settings[name]
        _save_settings_yaml(yaml_settings)
        _invalidate_config_cache()
        console.print(f"[green]Reset[/green] [cyan]{name}[/cyan] to env/default")
    else:
        console.print(f"[dim]{name} has no override in settings.yaml[/dim]")


@config_app.command("models")
def config_models() -> None:
    """Show configured LLM models."""
    from agent.config import get_models

    models = get_models()
    if not models:
        console.print("[dim]No custom models configured. Add them to ~/.devagent/settings.yaml[/dim]")
        console.print(
            "\n[dim]Example:\n"
            "models:\n"
            "  - name: my-gpt\n"
            "    provider: openai\n"
            "    model: gpt-4o\n"
            "    api_key: ${OPENAI_API_KEY}[/dim]"
        )
        return

    table = Table(title="LLM Models", show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Provider", style="bold")
    table.add_column("Model", style="bold")
    table.add_column("Base URL", style="dim")

    for m in models:
        table.add_row(m.name, m.provider, m.model, m.base_url or "default")

    console.print(table)


@config_app.command("node-models")
def config_node_models() -> None:
    """Show which model is assigned to each pipeline node."""
    from agent.config import get_node_models
    from agent.llm import _NODE_MODEL_MAP, resolve_model_name_for_node

    node_models = get_node_models()

    table = Table(title="Node → Model Mapping", show_lines=True)
    table.add_column("Node", style="cyan", no_wrap=True)
    table.add_column("Tier", style="dim")
    table.add_column("Resolved Model", style="bold")
    table.add_column("Override", style="dim")

    for node in sorted(_NODE_MODEL_MAP):
        tier = _NODE_MODEL_MAP[node]
        resolved_str = resolve_model_name_for_node(node)
        override = node_models.get(node, "")
        override_style = f"[yellow]{override}[/yellow]" if override else ""
        table.add_row(node, tier, resolved_str, override_style)

    if "default_strong" in node_models or "default_fast" in node_models:
        console.print()
        if "default_strong" in node_models:
            console.print(f"  default_strong = [yellow]{node_models['default_strong']}[/yellow]")
        if "default_fast" in node_models:
            console.print(f"  default_fast = [yellow]{node_models['default_fast']}[/yellow]")

    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
