"""Tool manager: static tool sets per node (Reader/Executor/Reviewer)."""

from __future__ import annotations


import structlog

log = structlog.get_logger()

STATIC_TOOLS: dict[str, list[str]] = {
    "reader": ["get_task", "get_comments"],
    "executor": ["write_file", "run_command", "create_branch", "commit"],
    "reviewer": ["run_tests", "run_linter", "run_type_check", "get_diff"],
}


def get_tools_for_node(node_name: str) -> list[str]:
    """Return the static tool set for a given node."""
    tools = STATIC_TOOLS.get(node_name, [])
    log.debug("tools_resolved", node=node_name, tools=tools)
    return tools
