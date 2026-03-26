"""Base MCP client: connection management + auto-discovery of tools.

Connects to any MCP server via stdio transport, discovers available tools,
and provides a generic call interface. Provider implementations use this
to map discovered tools to the Provider ABCs.
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

import structlog

from agent.errors import DegradedError, PermanentError, TransientError
from agent.utils.timeout import with_timeout

log = structlog.get_logger()


@dataclass
class ToolInfo:
    """Discovered MCP tool metadata."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class MCPConnection:
    """Active MCP server connection with discovered tools."""

    server_name: str
    tools: dict[str, ToolInfo] = field(default_factory=dict)
    session: Any = None
    _exit_stack: AsyncExitStack | None = None

    @property
    def tool_names(self) -> list[str]:
        return list(self.tools.keys())

    def has_tool(self, name: str) -> bool:
        return name in self.tools


class MCPClient:
    """Generic MCP client with auto-discovery.

    Usage:
        client = MCPClient()
        await client.connect("tracker", command="python", args=["tracker_server.py"])
        tools = client.get_tools("tracker")  # auto-discovered
        result = await client.call_tool("tracker", "get_issue", {"issue_id": "PROJ-1"})
    """

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}

    async def connect(
        self,
        server_name: str,
        *,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> MCPConnection:
        """Connect to an MCP server and auto-discover its tools.

        Args:
            server_name: Logical name for this connection (e.g. "tracker", "github")
            command: Command to launch the server (e.g. "python", "node", "npx")
            args: Arguments for the command (e.g. ["server.py"] or ["-y", "@mcp/github"])
            env: Optional environment variables for the server process
            timeout: Connection timeout in seconds
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise PermanentError(
                "MCP SDK not installed. Run: pip install mcp"
            )

        exit_stack = AsyncExitStack()

        try:
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env,
            )

            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            session = await exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )

            await with_timeout(
                session.initialize(),
                timeout=timeout,
                name=f"MCP init ({server_name})",
            )

            response = await with_timeout(
                session.list_tools(),
                timeout=timeout,
                name=f"MCP list_tools ({server_name})",
            )

            tools: dict[str, ToolInfo] = {}
            for tool in response.tools:
                tools[tool.name] = ToolInfo(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
                )

            conn = MCPConnection(
                server_name=server_name,
                tools=tools,
                session=session,
                _exit_stack=exit_stack,
            )
            self._connections[server_name] = conn

            log.info(
                "mcp_connected",
                server=server_name,
                tools=list(tools.keys()),
                count=len(tools),
            )
            return conn

        except TransientError:
            await exit_stack.aclose()
            raise
        except Exception as exc:
            await exit_stack.aclose()
            raise PermanentError(f"MCP connection to '{server_name}' failed: {exc}") from exc

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> Any:
        """Call a tool on a connected MCP server.

        Raises DegradedError if the tool is not available.
        """
        conn = self._connections.get(server_name)
        if conn is None:
            raise PermanentError(f"MCP server '{server_name}' not connected")

        if not conn.has_tool(tool_name):
            raise DegradedError(
                f"Tool '{tool_name}' not available on '{server_name}'. "
                f"Available: {conn.tool_names}"
            )

        try:
            result = await with_timeout(
                conn.session.call_tool(tool_name, arguments or {}),
                timeout=timeout,
                name=f"MCP {server_name}.{tool_name}",
            )
            return _parse_mcp_result(result)
        except (TransientError, DegradedError):
            raise
        except Exception as exc:
            raise TransientError(f"MCP call {server_name}.{tool_name} failed: {exc}") from exc

    async def call_tool_safe(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        default: Any = None,
        timeout: int = 30,
    ) -> Any:
        """Call a tool with graceful degradation — returns default on failure."""
        try:
            return await self.call_tool(server_name, tool_name, arguments, timeout)
        except (TransientError, DegradedError, PermanentError) as exc:
            log.warning(
                "mcp_call_degraded",
                server=server_name,
                tool=tool_name,
                error=str(exc),
            )
            return default

    def get_connection(self, server_name: str) -> MCPConnection | None:
        return self._connections.get(server_name)

    def get_tools(self, server_name: str) -> list[ToolInfo]:
        """Get auto-discovered tools for a server."""
        conn = self._connections.get(server_name)
        if conn is None:
            return []
        return list(conn.tools.values())

    def list_servers(self) -> list[str]:
        return list(self._connections.keys())

    async def disconnect(self, server_name: str) -> None:
        """Disconnect from a server and clean up resources."""
        conn = self._connections.pop(server_name, None)
        if conn and conn._exit_stack:
            await conn._exit_stack.aclose()
            log.info("mcp_disconnected", server=server_name)

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for name in list(self._connections.keys()):
            await self.disconnect(name)


def _parse_mcp_result(result: Any) -> Any:
    """Parse MCP tool result into a usable Python object."""
    if result is None:
        return None

    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list):
            texts = []
            for item in content:
                if hasattr(item, "text"):
                    texts.append(item.text)
                elif isinstance(item, str):
                    texts.append(item)
            combined = "\n".join(texts)
            try:
                return json.loads(combined)
            except (json.JSONDecodeError, ValueError):
                return combined
        if isinstance(content, str):
            try:
                return json.loads(content)
            except (json.JSONDecodeError, ValueError):
                return content
        return content

    return result
