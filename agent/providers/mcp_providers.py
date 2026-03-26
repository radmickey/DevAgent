"""MCP-based provider implementations with auto-discovery.

Each provider wraps an MCPClient connection and maps discovered tools
to the corresponding Provider ABC methods. Tool mapping is configurable
via TOOL_MAPPING but auto-discovered by default using heuristic name matching.
"""

from __future__ import annotations

from typing import Any

import structlog

from agent.errors import DegradedError
from agent.providers.base import CodeProvider, DocProvider, TaskProvider
from agent.providers.mcp_client import MCPClient, MCPConnection

log = structlog.get_logger()


# Default tool name patterns → provider methods.
# Users can override via MCP_TOOL_MAPPING env / config.
_TASK_TOOL_HINTS = {
    "get_task": ["get_task", "get_issue", "getIssue", "read_issue", "fetch_task"],
    "get_comments": ["get_comments", "list_comments", "getComments", "get_issue_comments"],
    "update_status": ["update_status", "transition_issue", "updateStatus", "set_status"],
}

_CODE_TOOL_HINTS = {
    "search_code": ["search_code", "searchCode", "code_search", "grep", "search"],
    "get_file": ["get_file", "getFile", "read_file", "get_file_contents", "getFileContents"],
    "list_files": ["list_files", "listFiles", "list_directory", "listDirectory", "ls"],
    "get_diff": ["get_diff", "getDiff", "compare", "diff"],
}

_DOC_TOOL_HINTS = {
    "search_docs": ["search_docs", "searchDocs", "search", "search_pages", "query"],
    "get_page": ["get_page", "getPage", "read_page", "get_document", "getDocument"],
}


def _resolve_tool(
    conn: MCPConnection,
    hints: list[str],
    custom_mapping: dict[str, str] | None = None,
    method_name: str = "",
) -> str | None:
    """Find the best matching MCP tool for a provider method.

    Priority: custom_mapping > exact hint match > substring match.
    """
    if custom_mapping and method_name in custom_mapping:
        candidate = custom_mapping[method_name]
        if conn.has_tool(candidate):
            return candidate

    for hint in hints:
        if conn.has_tool(hint):
            return hint

    for tool_name in conn.tool_names:
        lower = tool_name.lower()
        for hint in hints:
            if hint.lower() in lower or lower in hint.lower():
                return tool_name

    return None


class MCPTaskProvider(TaskProvider):
    """Task provider backed by an MCP server.

    Auto-discovers tools for get_task, get_comments, update_status
    by matching against known tool name patterns.
    """

    def __init__(
        self,
        client: MCPClient,
        server_name: str,
        tool_mapping: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self._server = server_name
        self._mapping = tool_mapping
        self._resolved: dict[str, str | None] = {}

    def _resolve(self, method: str) -> str | None:
        if method not in self._resolved:
            conn = self._client.get_connection(self._server)
            if conn is None:
                return None
            hints = _TASK_TOOL_HINTS.get(method, [])
            self._resolved[method] = _resolve_tool(conn, hints, self._mapping, method)
        return self._resolved[method]

    async def get_task(self, task_id: str) -> dict[str, Any]:
        tool = self._resolve("get_task")
        if tool is None:
            raise DegradedError(
                f"No get_task tool found on MCP server '{self._server}'"
            )
        result = await self._client.call_tool(
            self._server, tool, {"task_id": task_id}
        )
        if isinstance(result, dict):
            return result
        return {"id": task_id, "raw": result}

    async def get_comments(self, task_id: str) -> list[dict[str, Any]]:
        tool = self._resolve("get_comments")
        if tool is None:
            log.debug("mcp_no_tool", server=self._server, method="get_comments")
            return []
        result = await self._client.call_tool_safe(
            self._server, tool, {"task_id": task_id}, default=[]
        )
        if isinstance(result, list):
            return result
        return []

    async def update_status(self, task_id: str, status: str) -> None:
        tool = self._resolve("update_status")
        if tool is None:
            log.warning("mcp_no_tool", server=self._server, method="update_status")
            return
        await self._client.call_tool_safe(
            self._server, tool, {"task_id": task_id, "status": status}
        )


class MCPCodeProvider(CodeProvider):
    """Code provider backed by an MCP server.

    Auto-discovers tools for search_code, get_file, list_files, get_diff.
    """

    def __init__(
        self,
        client: MCPClient,
        server_name: str,
        tool_mapping: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self._server = server_name
        self._mapping = tool_mapping
        self._resolved: dict[str, str | None] = {}

    def _resolve(self, method: str) -> str | None:
        if method not in self._resolved:
            conn = self._client.get_connection(self._server)
            if conn is None:
                return None
            hints = _CODE_TOOL_HINTS.get(method, [])
            self._resolved[method] = _resolve_tool(conn, hints, self._mapping, method)
        return self._resolved[method]

    async def search_code(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        tool = self._resolve("search_code")
        if tool is None:
            raise DegradedError(
                f"No search_code tool found on MCP server '{self._server}'"
            )
        result = await self._client.call_tool(
            self._server, tool, {"query": query, "limit": limit}
        )
        if isinstance(result, list):
            return result
        return [{"raw": result}]

    async def get_file(self, path: str, ref: str = "main") -> str:
        tool = self._resolve("get_file")
        if tool is None:
            raise DegradedError(
                f"No get_file tool found on MCP server '{self._server}'"
            )
        result = await self._client.call_tool(
            self._server, tool, {"path": path, "ref": ref}
        )
        if isinstance(result, str):
            return result
        return str(result)

    async def list_files(self, path: str = "", ref: str = "main") -> list[str]:
        tool = self._resolve("list_files")
        if tool is None:
            raise DegradedError(
                f"No list_files tool found on MCP server '{self._server}'"
            )
        result = await self._client.call_tool(
            self._server, tool, {"path": path, "ref": ref}
        )
        if isinstance(result, list):
            return [str(f) for f in result]
        return []

    async def get_diff(self, base: str, head: str) -> str:
        tool = self._resolve("get_diff")
        if tool is None:
            raise DegradedError(
                f"No get_diff tool found on MCP server '{self._server}'"
            )
        result = await self._client.call_tool(
            self._server, tool, {"base": base, "head": head}
        )
        return str(result) if result else ""


class MCPDocProvider(DocProvider):
    """Doc provider backed by an MCP server.

    Auto-discovers tools for search_docs, get_page.
    """

    def __init__(
        self,
        client: MCPClient,
        server_name: str,
        tool_mapping: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self._server = server_name
        self._mapping = tool_mapping
        self._resolved: dict[str, str | None] = {}

    def _resolve(self, method: str) -> str | None:
        if method not in self._resolved:
            conn = self._client.get_connection(self._server)
            if conn is None:
                return None
            hints = _DOC_TOOL_HINTS.get(method, [])
            self._resolved[method] = _resolve_tool(conn, hints, self._mapping, method)
        return self._resolved[method]

    async def search_docs(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        tool = self._resolve("search_docs")
        if tool is None:
            raise DegradedError(
                f"No search_docs tool found on MCP server '{self._server}'"
            )
        result = await self._client.call_tool(
            self._server, tool, {"query": query, "limit": limit}
        )
        if isinstance(result, list):
            return result
        return [{"raw": result}]

    async def get_page(self, page_id: str) -> dict[str, Any]:
        tool = self._resolve("get_page")
        if tool is None:
            raise DegradedError(
                f"No get_page tool found on MCP server '{self._server}'"
            )
        result = await self._client.call_tool(
            self._server, tool, {"page_id": page_id}
        )
        if isinstance(result, dict):
            return result
        return {"id": page_id, "raw": result}
