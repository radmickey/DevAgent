"""Shared fixtures for DevAgent tests."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent.providers.base import TaskProvider, CodeProvider, DocProvider
from agent.pipeline.state import PipelineState


@pytest.fixture
def mock_task_provider():
    provider = AsyncMock(spec=TaskProvider)
    provider.get_task.return_value = {
        "id": "TEST-1",
        "title": "Test task",
        "description": "Test description",
        "comments": [],
        "labels": [],
        "status": "open",
    }
    provider.get_comments.return_value = []
    return provider


@pytest.fixture
def mock_code_provider():
    provider = AsyncMock(spec=CodeProvider)
    provider.search_code.return_value = []
    provider.get_file.return_value = ""
    provider.list_files.return_value = []
    provider.get_diff.return_value = ""
    return provider


@pytest.fixture
def mock_doc_provider():
    provider = AsyncMock(spec=DocProvider)
    provider.search_docs.return_value = []
    provider.get_page.return_value = {"id": "1", "title": "Test", "content": ""}
    return provider


@pytest.fixture
def base_state() -> PipelineState:
    return PipelineState(
        task_id="TEST-1",
        task_raw={},
        enriched_context={},
        ranked_context=[],
        plan=None,
        plan_approved=False,
        human_feedback="",
        code_changes=[],
        review_result={},
        has_code_context=False,
        has_doc_context=False,
        has_similar_tasks=False,
        enrichment_warnings=[],
    )
