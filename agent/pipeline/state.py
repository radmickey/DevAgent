"""PipelineState TypedDict — the single state object flowing through the graph."""

from __future__ import annotations

from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    task_id: str
    task_raw: dict[str, Any]
    enriched_context: dict[str, Any]
    ranked_context: list[dict[str, Any]]
    has_code_context: bool
    has_doc_context: bool
    has_similar_tasks: bool
    enrichment_warnings: list[str]
    plan: Any
    plan_approved: bool
    human_feedback: str
    code_changes: list[dict[str, Any]]
    review_result: dict[str, Any]
    iteration_count: int
    dry_run: bool
    contracts: list[dict[str, Any]]
    contract_violations: list[dict[str, Any]]
    detected_languages: list[str]
