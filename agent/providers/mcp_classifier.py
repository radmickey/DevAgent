"""MCP tool classifier: rule-based by default, LLM as optional enhancement.

Classifies tools into pipeline stages. Each tool can belong to MULTIPLE stages.
Tools that don't match any stage get SKIP (excluded from pipeline).

Flow:
  1. Rule-based classification (always works, no dependencies)
  2. If LLM is available and configured: refine with one fast LLM call
  3. Apply user overrides from YAML config
  4. Cache results
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

import structlog

from agent.config import DEVAGENT_HOME
from agent.providers.mcp_client import ToolInfo

log = structlog.get_logger()

CACHE_PATH = DEVAGENT_HOME / "tool_classifications.json"


class ToolStage(str, Enum):
    """Pipeline stages a tool can be assigned to."""

    CONTEXT_GATHERING = "context_gathering"
    PLANNING = "planning"
    CODE_OPERATIONS = "code_operations"
    REVIEW = "review"
    TASK_MANAGEMENT = "task_management"
    GENERIC = "generic"
    SKIP = "skip"


_STAGE_KEYWORDS: dict[ToolStage, list[tuple[str, int]]] = {
    ToolStage.TASK_MANAGEMENT: [
        ("issue", 3), ("task", 3), ("ticket", 3), ("story", 3), ("bug", 3),
        ("sprint", 2), ("transition", 2), ("assign", 2), ("status", 2),
        ("comment", 1), ("label", 1),
    ],
    ToolStage.CODE_OPERATIONS: [
        ("write_file", 5), ("create_file", 5), ("edit_file", 5), ("delete_file", 5),
        ("commit", 4), ("push", 3), ("pull_request", 4), ("merge", 3),
        ("branch", 2), ("mkdir", 2), ("run_command", 3), ("execute", 2), ("shell", 2),
        ("create_or_update", 3),
    ],
    ToolStage.REVIEW: [
        ("test", 3), ("lint", 3), ("check_types", 3), ("analyze", 2),
        ("review", 2), ("validate", 2), ("verify", 2), ("audit", 2),
        ("coverage", 2), ("benchmark", 1),
    ],
    ToolStage.CONTEXT_GATHERING: [
        ("search", 2), ("find", 2), ("query", 2), ("list", 1), ("get", 1),
        ("fetch", 1), ("read", 1), ("browse", 1), ("lookup", 1), ("grep", 2),
    ],
    ToolStage.PLANNING: [
        ("plan", 3), ("requirement", 3), ("depend", 2), ("architecture", 2),
        ("design", 2), ("estimate", 2), ("scope", 2),
    ],
}

# Multi-stage: tools matching these patterns belong to MULTIPLE stages.
# E.g. "search_code" is useful in both context_gathering AND code_operations.
_MULTI_STAGE_RULES: list[tuple[list[str], list[ToolStage]]] = [
    (
        ["search_code", "search_files", "grep", "find_in_files"],
        [ToolStage.CONTEXT_GATHERING, ToolStage.CODE_OPERATIONS],
    ),
    (
        ["get_file", "read_file", "get_file_contents"],
        [ToolStage.CONTEXT_GATHERING, ToolStage.CODE_OPERATIONS, ToolStage.REVIEW],
    ),
    (
        ["list_files", "list_directory"],
        [ToolStage.CONTEXT_GATHERING, ToolStage.CODE_OPERATIONS],
    ),
    (
        ["get_issue", "get_task"],
        [ToolStage.TASK_MANAGEMENT, ToolStage.CONTEXT_GATHERING],
    ),
    (
        ["search_docs", "search_pages"],
        [ToolStage.CONTEXT_GATHERING, ToolStage.PLANNING],
    ),
]

CLASSIFICATION_PROMPT = """\
You are a tool classifier for a software development AI agent pipeline.
The pipeline has these stages:
- context_gathering: enrichment tools that search/fetch context (search code, docs, databases)
- planning: tools that help create execution plans (requirements, dependencies, architecture)
- code_operations: tools that read/write/modify code or run commands
- review: tools that test/lint/analyze code quality
- task_management: tools that manage tasks/issues
- skip: tools irrelevant to software development (e.g. weather, music, calendar)

A tool CAN belong to MULTIPLE stages. For example, "search_code" belongs to both \
context_gathering and code_operations.

Tools to classify:
{tools_text}

Respond with ONLY a JSON object mapping tool names to a LIST of stages. Example:
{{"get_issue": ["task_management", "context_gathering"], \
"search_code": ["context_gathering", "code_operations"], \
"run_tests": ["review"], \
"play_music": ["skip"]}}
"""


def _compute_tools_hash(tools: list[ToolInfo]) -> str:
    """Deterministic hash of tool names to detect changes."""
    names = sorted(t.name for t in tools)
    return hashlib.sha256("|".join(names).encode()).hexdigest()[:16]


def _load_cache(tools_hash: str) -> dict[str, list[str]] | None:
    """Load cached classifications if hash matches."""
    if not CACHE_PATH.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(CACHE_PATH.read_text())
        if data.get("hash") == tools_hash:
            result: dict[str, list[str]] = data.get("classifications", {})
            return result
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_cache(tools_hash: str, classifications: dict[str, list[str]]) -> None:
    """Persist classifications to disk."""
    try:
        CACHE_PATH.write_text(
            json.dumps({"hash": tools_hash, "classifications": classifications}, indent=2)
        )
    except OSError as exc:
        log.warning("classifier_cache_write_failed", error=str(exc))


def _format_tools_for_prompt(tools: list[ToolInfo]) -> str:
    lines: list[str] = []
    for t in tools:
        desc = t.description[:200] if t.description else "(no description)"
        lines.append(f"- {t.name}: {desc}")
    return "\n".join(lines)


async def classify_tools(
    tools: list[ToolInfo],
    *,
    use_cache: bool = True,
    use_llm: bool = True,
    user_overrides: dict[str, list[str]] | None = None,
    exclude_tools: list[str] | None = None,
) -> dict[str, list[ToolStage]]:
    """Classify MCP tools by pipeline stage.

    Each tool maps to a LIST of stages (multi-stage support).
    Always starts with rule-based classification, LLM refines if available.

    Args:
        tools: Discovered MCP tools
        use_cache: Check file cache before classifying
        use_llm: Try LLM classification (falls back to rules if unavailable)
        user_overrides: Manual stage assignments from YAML config
        exclude_tools: Tool names to skip entirely
    """
    if not tools:
        return {}

    tools_hash = _compute_tools_hash(tools)

    if use_cache:
        cached = _load_cache(tools_hash)
        if cached is not None:
            log.info("classifier_cache_hit", hash=tools_hash, count=len(cached))
            result = {name: [_parse_stage(s) for s in stages] for name, stages in cached.items()}
            return _apply_overrides(result, user_overrides, exclude_tools)

    classifications = _classify_by_rules(tools)

    if use_llm:
        try:
            llm_result = await _call_llm_classifier(tools)
            classifications = _merge_classifications(classifications, llm_result)
        except Exception as exc:
            log.info("classifier_llm_skipped", reason=str(exc))

    classifications = _apply_overrides(classifications, user_overrides, exclude_tools)

    raw_cache = {
        name: [s.value for s in stages]
        for name, stages in classifications.items()
    }
    _save_cache(tools_hash, raw_cache)

    log.info(
        "classifier_done",
        total=len(classifications),
        multi_stage=sum(1 for s in classifications.values() if len(s) > 1),
        skipped=sum(1 for s in classifications.values() if ToolStage.SKIP in s),
    )

    return classifications


def _classify_by_rules(tools: list[ToolInfo]) -> dict[str, list[ToolStage]]:
    """Rule-based classification. Always available, no dependencies."""
    result: dict[str, list[ToolStage]] = {}
    for tool in tools:
        stages = _classify_single_by_rules(tool)
        result[tool.name] = stages
    return result


def _classify_single_by_rules(tool: ToolInfo) -> list[ToolStage]:
    """Classify a single tool by rules. Returns multiple stages if applicable."""
    tool_lower = tool.name.lower()

    for patterns, stages in _MULTI_STAGE_RULES:
        if any(p in tool_lower for p in patterns):
            return list(stages)

    text = f"{tool.name} {tool.description}".lower()

    scores: dict[ToolStage, int] = {}
    for stage, keywords in _STAGE_KEYWORDS.items():
        score = sum(weight for kw, weight in keywords if kw in text)
        if score > 0:
            scores[stage] = score

    if not scores:
        return [ToolStage.SKIP]

    max_score = max(scores.values())
    threshold = max(1, max_score - 2)
    return [stage for stage, score in scores.items() if score >= threshold]


async def _call_llm_classifier(tools: list[ToolInfo]) -> dict[str, list[ToolStage]]:
    """Call the fast LLM to classify tools. Returns multi-stage mapping."""
    tools_text = _format_tools_for_prompt(tools)
    prompt = CLASSIFICATION_PROMPT.format(tools_text=tools_text)

    from pydantic_ai import Agent

    from agent.pipeline.prompts import get_prompt

    classifier_agent = Agent(
        "test",
        output_type=str,
        system_prompt=get_prompt("classifier"),
    )

    from agent.llm import get_fast_model
    model = get_fast_model()

    log.info("classifier_calling_llm", tool_count=len(tools))
    result = await classifier_agent.run(prompt, model=model)
    raw_text = result.output.strip()

    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    parsed: dict[str, Any] = json.loads(raw_text)

    classifications: dict[str, list[ToolStage]] = {}
    for tool in tools:
        raw_stages = parsed.get(tool.name, ["generic"])
        if isinstance(raw_stages, str):
            raw_stages = [raw_stages]
        classifications[tool.name] = [_parse_stage(s) for s in raw_stages]

    log.info("classifier_llm_done", classified=len(classifications))
    return classifications


def _merge_classifications(
    rules: dict[str, list[ToolStage]],
    llm: dict[str, list[ToolStage]],
) -> dict[str, list[ToolStage]]:
    """Merge rule-based and LLM classifications: union of stages."""
    merged: dict[str, list[ToolStage]] = {}
    all_tools = set(rules) | set(llm)
    for name in all_tools:
        stages_set: set[ToolStage] = set()
        stages_set.update(rules.get(name, []))
        stages_set.update(llm.get(name, []))
        stages_set.discard(ToolStage.SKIP)
        if not stages_set:
            stages_set = {ToolStage.SKIP}
        merged[name] = sorted(stages_set, key=lambda s: s.value)
    return merged


def _apply_overrides(
    classifications: dict[str, list[ToolStage]],
    user_overrides: dict[str, list[str]] | None,
    exclude_tools: list[str] | None,
) -> dict[str, list[ToolStage]]:
    """Apply user overrides from YAML config."""
    if exclude_tools:
        for tool_name in exclude_tools:
            if tool_name in classifications:
                classifications[tool_name] = [ToolStage.SKIP]

    if user_overrides:
        for tool_name, stages_raw in user_overrides.items():
            if tool_name in classifications:
                classifications[tool_name] = [_parse_stage(s) for s in stages_raw]

    return classifications


def _parse_stage(value: str) -> ToolStage:
    """Safely parse a stage string into ToolStage enum."""
    try:
        return ToolStage(value)
    except ValueError:
        return ToolStage.GENERIC


# Keep backward-compat aliases
def _fallback_classify(tools: list[ToolInfo]) -> dict[str, list[ToolStage]]:
    """Rule-based fallback when LLM is unavailable."""
    return _classify_by_rules(tools)


def _classify_by_heuristic(tool: ToolInfo) -> ToolStage:
    """Backward-compat: returns the primary stage for a tool."""
    stages = _classify_single_by_rules(tool)
    return stages[0] if stages else ToolStage.GENERIC
