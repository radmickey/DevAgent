"""ToolCatalog: unified registry of classified MCP tools.

Pipeline nodes query this catalog to get tools relevant to their stage.
Supports multi-stage tools (a tool can appear in multiple stages).
Tools classified as SKIP are excluded from the pipeline entirely.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog

from agent.providers.mcp_classifier import ToolStage
from agent.providers.mcp_client import MCPClient, ToolInfo

log = structlog.get_logger()


@dataclass
class CatalogTool:
    """A classified MCP tool with full metadata."""

    name: str
    server: str
    description: str
    input_schema: dict[str, Any]
    stages: list[ToolStage]

    @property
    def stage(self) -> ToolStage:
        """Primary stage (first in list). Backward compat."""
        return self.stages[0] if self.stages else ToolStage.GENERIC


@dataclass
class ToolCall:
    """A pending tool call for parallel execution."""

    server: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


class ToolCatalog:
    """Central registry of all classified MCP tools across all servers.

    Supports multi-stage tools: a single tool can appear in multiple pipeline
    stages. Tools classified as SKIP are silently excluded.
    """

    def __init__(self, client: MCPClient) -> None:
        self._client = client
        self._tools: dict[str, CatalogTool] = {}
        self._by_stage: dict[ToolStage, list[CatalogTool]] = {s: [] for s in ToolStage}
        self._skipped: list[str] = []

    def register(
        self,
        *,
        server: str,
        tool: ToolInfo,
        stages: list[ToolStage],
    ) -> None:
        """Register a tool in the catalog, potentially in multiple stages."""
        if not stages or stages == [ToolStage.SKIP]:
            self._skipped.append(f"{server}::{tool.name}")
            log.debug("tool_skipped", server=server, tool=tool.name)
            return

        key = f"{server}::{tool.name}"
        ct = CatalogTool(
            name=tool.name,
            server=server,
            description=tool.description,
            input_schema=tool.input_schema,
            stages=stages,
        )
        self._tools[key] = ct

        for stage in stages:
            if stage != ToolStage.SKIP:
                self._by_stage[stage].append(ct)

    def register_batch(
        self,
        server: str,
        tools: list[ToolInfo],
        classifications: dict[str, list[ToolStage]],
    ) -> None:
        """Register multiple tools from a server with their classifications."""
        for tool in tools:
            stages = classifications.get(tool.name, [ToolStage.GENERIC])
            self.register(server=server, tool=tool, stages=stages)

    def get_tools_for_stage(self, stage: ToolStage | str) -> list[CatalogTool]:
        """Get all tools assigned to a pipeline stage.

        Also includes GENERIC tools which are available everywhere.
        """
        if isinstance(stage, str):
            try:
                stage = ToolStage(stage)
            except ValueError:
                return []

        result = list(self._by_stage.get(stage, []))
        if stage not in (ToolStage.GENERIC, ToolStage.SKIP):
            result.extend(self._by_stage.get(ToolStage.GENERIC, []))

        seen: set[str] = set()
        deduped: list[CatalogTool] = []
        for t in result:
            key = f"{t.server}::{t.name}"
            if key not in seen:
                seen.add(key)
                deduped.append(t)
        return deduped

    def get_all_tools(self) -> list[CatalogTool]:
        """Get all registered tools (excludes SKIP)."""
        return list(self._tools.values())

    def get_skipped_tools(self) -> list[str]:
        """Get names of tools that were excluded."""
        return list(self._skipped)

    def has_tools_for_stage(self, stage: ToolStage | str) -> bool:
        """Check if any tools are available for a stage."""
        return len(self.get_tools_for_stage(stage)) > 0

    async def call_tool(
        self,
        server: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> Any:
        """Call a tool via the underlying MCPClient."""
        return await self._client.call_tool(server, tool_name, arguments, timeout)

    async def call_tool_safe(
        self,
        server: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        default: Any = None,
        timeout: int = 30,
    ) -> Any:
        """Call a tool with graceful degradation."""
        return await self._client.call_tool_safe(
            server, tool_name, arguments, default, timeout
        )

    async def call_tools_parallel(
        self,
        calls: list[ToolCall],
        timeout: int = 30,
    ) -> list[Any]:
        """Execute multiple tool calls in parallel with graceful degradation."""
        coros = [
            self._client.call_tool_safe(
                c.server, c.tool, c.arguments, default=None, timeout=timeout,
            )
            for c in calls
        ]
        return list(await asyncio.gather(*coros, return_exceptions=True))

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def skipped_count(self) -> int:
        return len(self._skipped)

    def summary(self) -> dict[str, int]:
        """Return a count of tools per stage."""
        result = {
            stage.value: len(tools)
            for stage, tools in self._by_stage.items()
            if tools
        }
        if self._skipped:
            result["skip"] = len(self._skipped)
        return result
