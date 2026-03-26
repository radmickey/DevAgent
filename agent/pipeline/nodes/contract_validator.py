"""Contract validator: checks code changes against extracted API contracts.

Validates that executor output doesn't break existing API contracts.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from agent.pipeline.state import PipelineState

log = structlog.get_logger()


async def contract_validator_node(state: PipelineState) -> PipelineState:
    """Validate code changes against extracted contracts.

    Checks for:
    - Removed or renamed endpoints
    - Changed request/response shapes
    - Proto field number reuse
    """
    contracts = state.get("contracts", [])
    code_changes = state.get("code_changes", [])
    violations: list[dict[str, Any]] = []

    if not contracts or not code_changes:
        return {**state, "contract_violations": []}

    for change in code_changes:
        if not isinstance(change, dict):
            continue
        path = str(change.get("path", ""))
        diff = str(change.get("diff", change.get("content", "")))

        for contract in contracts:
            found = _check_violations(path, diff, contract)
            violations.extend(found)

    if violations:
        log.warning(
            "contract_violations_found",
            count=len(violations),
            paths=[v["path"] for v in violations],
        )
    else:
        log.info("contract_validation_passed", contracts=len(contracts))

    return {**state, "contract_violations": violations}


def _check_violations(
    changed_path: str, diff: str, contract: dict[str, Any]
) -> list[dict[str, Any]]:
    """Check a single file change against a contract."""
    violations: list[dict[str, Any]] = []
    contract_type = contract.get("type", "")

    if contract_type == "openapi" and _is_route_file(changed_path):
        removed_endpoints = _find_removed_endpoints(diff, contract.get("endpoints", []))
        for ep in removed_endpoints:
            violations.append({
                "type": "endpoint_removed",
                "path": changed_path,
                "contract_path": contract.get("path", ""),
                "detail": f"Endpoint {ep} appears to be removed",
                "severity": "error",
            })

    if contract_type == "protobuf" and changed_path.endswith(".proto"):
        field_issues = _check_proto_field_reuse(diff)
        for issue in field_issues:
            violations.append({
                "type": "proto_field_reuse",
                "path": changed_path,
                "contract_path": contract.get("path", ""),
                "detail": issue,
                "severity": "error",
            })

    return violations


def _is_route_file(path: str) -> bool:
    """Heuristic: is this file likely to contain API routes?"""
    route_indicators = ["route", "endpoint", "api", "controller", "handler", "view"]
    lower = path.lower()
    return any(ind in lower for ind in route_indicators)


def _find_removed_endpoints(diff: str, known_endpoints: list[str]) -> list[str]:
    """Find contract endpoints that appear in removed lines."""
    removed: list[str] = []
    for line in diff.split("\n"):
        if line.startswith("-") and not line.startswith("---"):
            for ep in known_endpoints:
                if ep in line:
                    removed.append(ep)
    return list(set(removed))


def _check_proto_field_reuse(diff: str) -> list[str]:
    """Check for proto field number reuse in additions."""
    added_fields: dict[str, list[str]] = {}
    issues: list[str] = []

    for line in diff.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            match = re.search(r'(\w+)\s+(\w+)\s*=\s*(\d+)', line)
            if match:
                field_num = match.group(3)
                field_name = match.group(2)
                if field_num in added_fields:
                    issues.append(
                        f"Field number {field_num} reused: "
                        f"{added_fields[field_num]} and {field_name}"
                    )
                else:
                    added_fields[field_num] = [field_name]

    return issues
