"""MCP bootstrap: connect → discover → classify → catalog.

Reads MCP server configs from YAML / env, connects to each one,
auto-discovers tools via list_tools(), classifies them by pipeline stage,
and builds a ToolCatalog for injection into pipeline nodes.
"""

from __future__ import annotations

import structlog

from agent.config import Config, MCPServerConfig, get_config
from agent.providers.mcp_classifier import classify_tools
from agent.providers.mcp_client import MCPClient, ToolInfo
from agent.providers.tool_catalog import ToolCatalog

log = structlog.get_logger()

_global_mcp_client: MCPClient | None = None
_global_catalog: ToolCatalog | None = None


def get_mcp_client() -> MCPClient:
    """Get the global MCPClient singleton (creates if needed)."""
    global _global_mcp_client
    if _global_mcp_client is None:
        _global_mcp_client = MCPClient()
    return _global_mcp_client


def get_tool_catalog() -> ToolCatalog | None:
    """Get the global ToolCatalog (None if bootstrap hasn't run)."""
    return _global_catalog


async def bootstrap_mcp(config: Config | None = None) -> ToolCatalog:
    """Connect to all MCP servers, classify tools, build catalog.

    Flow:
        1. Load server configs from YAML + env
        2. Connect to each server (discover tools)
        3. Collect user overrides / excludes from config
        4. Classify via rules + optional LLM
        5. Build ToolCatalog
        6. Return catalog (pipeline auto-resolves providers from it)
    """
    global _global_catalog

    if config is None:
        config = get_config()

    client = get_mcp_client()
    servers: list[MCPServerConfig] = config.mcp_servers

    if not servers:
        log.debug("mcp_bootstrap_skip", reason="no MCP servers configured")
        catalog = ToolCatalog(client)
        _global_catalog = catalog
        return catalog

    all_tools: list[tuple[str, list[ToolInfo]]] = []

    for server_cfg in servers:
        try:
            conn = await client.connect(
                server_cfg.name,
                command=server_cfg.command,
                args=server_cfg.args,
                env=server_cfg.env,
            )
            tools = list(conn.tools.values())
            all_tools.append((server_cfg.name, tools))

            log.info(
                "mcp_server_connected",
                server=server_cfg.name,
                tools=[t.name for t in tools],
            )
        except Exception as exc:
            log.error(
                "mcp_bootstrap_connect_failed",
                server=server_cfg.name,
                error=str(exc),
            )

    all_excludes: list[str] = []
    all_overrides: dict[str, list[str]] = {}
    server_map = {s.name: s for s in servers}
    for server_name, tools in all_tools:
        cfg = server_map.get(server_name)
        if cfg and cfg.exclude_tools:
            all_excludes.extend(cfg.exclude_tools)
        if cfg and cfg.tool_stages:
            all_overrides.update(cfg.tool_stages)

    flat_tools = [t for _, tools in all_tools for t in tools]
    classifications = await classify_tools(
        flat_tools,
        user_overrides=all_overrides or None,
        exclude_tools=all_excludes or None,
    )

    catalog = ToolCatalog(client)
    for server_name, tools in all_tools:
        catalog.register_batch(server_name, tools, classifications)

    _global_catalog = catalog

    log.info(
        "mcp_bootstrap_complete",
        servers=len(all_tools),
        total_tools=catalog.tool_count,
        skipped=catalog.skipped_count,
        by_stage=catalog.summary(),
    )

    return catalog


async def shutdown_mcp() -> None:
    """Disconnect all MCP servers and clean up."""
    global _global_mcp_client, _global_catalog
    if _global_mcp_client is not None:
        await _global_mcp_client.disconnect_all()
        _global_mcp_client = None
    _global_catalog = None
    log.info("mcp_shutdown_complete")
