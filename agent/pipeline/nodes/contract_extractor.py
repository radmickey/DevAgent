"""Contract extractor: discovers API contracts (OpenAPI, Proto, GraphQL) from the codebase.

Seeds the enricher with contract data for multi-language context.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from agent.pipeline.state import PipelineState
from agent.providers.base import CodeProvider

log = structlog.get_logger()

CONTRACT_PATTERNS: dict[str, list[str]] = {
    "openapi": ["openapi.yaml", "openapi.yml", "openapi.json", "swagger.yaml", "swagger.json"],
    "protobuf": ["*.proto"],
    "graphql": ["schema.graphql", "*.graphql", "schema.gql"],
    "asyncapi": ["asyncapi.yaml", "asyncapi.yml"],
}

_EXTENSION_TO_TYPE: dict[str, str] = {
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".gql": "graphql",
}


async def contract_extractor_node(
    state: PipelineState,
    *,
    code_provider: CodeProvider | None = None,
) -> PipelineState:
    """Extract API contracts from the codebase.

    Scans for OpenAPI/Proto/GraphQL files and extracts endpoint/service definitions.
    """
    task_id = state.get("task_id", "")
    contracts: list[dict[str, Any]] = []
    languages: list[str] = list(state.get("detected_languages", []))

    if code_provider is None:
        log.info("contract_extractor_no_provider", task_id=task_id)
        return {**state, "contracts": contracts, "detected_languages": languages}

    try:
        files = await code_provider.list_files("")
        contract_files = _find_contract_files(files)

        for cf in contract_files:
            try:
                content = await code_provider.get_file(cf["path"])
                extracted = _extract_contract_info(cf["path"], cf["type"], content)
                if extracted:
                    contracts.append(extracted)
            except Exception as exc:
                log.warning("contract_read_failed", path=cf["path"], error=str(exc))

        log.info(
            "contracts_extracted",
            task_id=task_id,
            count=len(contracts),
            types=[c["type"] for c in contracts],
        )
    except Exception as exc:
        log.warning("contract_extraction_failed", task_id=task_id, error=str(exc))

    return {**state, "contracts": contracts, "detected_languages": languages}


def _find_contract_files(files: list[str]) -> list[dict[str, str]]:
    """Identify contract files from a file listing."""
    found: list[dict[str, str]] = []
    for f in files:
        lower = f.lower()
        for contract_type, patterns in CONTRACT_PATTERNS.items():
            for pattern in patterns:
                if pattern.startswith("*"):
                    if lower.endswith(pattern[1:]):
                        found.append({"path": f, "type": contract_type})
                        break
                elif lower.endswith(pattern) or f"/{pattern}" in lower:
                    found.append({"path": f, "type": contract_type})
                    break
    return found


def _extract_contract_info(path: str, contract_type: str, content: str) -> dict[str, Any] | None:
    """Extract structured info from a contract file."""
    if not content.strip():
        return None

    info: dict[str, Any] = {
        "path": path,
        "type": contract_type,
        "endpoints": [],
        "services": [],
        "summary": "",
    }

    if contract_type == "openapi":
        info["endpoints"] = _extract_openapi_endpoints(content)
        info["summary"] = f"OpenAPI spec with {len(info['endpoints'])} endpoints"
    elif contract_type == "protobuf":
        info["services"] = _extract_proto_services(content)
        info["summary"] = f"Protobuf with {len(info['services'])} services"
    elif contract_type == "graphql":
        info["endpoints"] = _extract_graphql_types(content)
        info["summary"] = f"GraphQL with {len(info['endpoints'])} types"
    else:
        info["summary"] = f"{contract_type} contract at {path}"

    return info


def _extract_openapi_endpoints(content: str) -> list[str]:
    """Extract endpoint paths from OpenAPI YAML/JSON."""
    endpoints = []
    for match in re.finditer(r'^\s*(/[a-zA-Z0-9/_\-{}]+):', content, re.MULTILINE):
        endpoints.append(match.group(1))
    return endpoints


def _extract_proto_services(content: str) -> list[str]:
    """Extract service names from .proto files."""
    return re.findall(r'service\s+(\w+)\s*\{', content)


def _extract_graphql_types(content: str) -> list[str]:
    """Extract type names from GraphQL schema."""
    return re.findall(r'type\s+(\w+)\s*[\{\(]', content)
