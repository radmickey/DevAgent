"""Tests for pipeline nodes: input_router, reader, enricher, ranker, stubs."""

from __future__ import annotations

import pytest

from agent.pipeline.nodes.input_router import input_router_node
from agent.pipeline.nodes.reader import reader_node
from agent.pipeline.nodes.enricher import enricher_node
from agent.pipeline.nodes.ranker import ranker_node
from agent.pipeline.nodes.explainer import explainer_node
from agent.pipeline.nodes.executor import executor_node
from agent.pipeline.nodes.reviewer import reviewer_node
from agent.pipeline.nodes.doc_writer import doc_writer_node
from agent.errors import PermanentError


class TestInputRouter:
    @pytest.mark.asyncio
    async def test_routes_task_id(self, base_state):
        state = {**base_state, "task_id": "PROJ-123"}
        result = await input_router_node(state)
        assert result["task_id"] == "PROJ-123"
        assert result["task_raw"] == {}

    @pytest.mark.asyncio
    async def test_routes_numeric_task_id(self, base_state):
        state = {**base_state, "task_id": "#456"}
        result = await input_router_node(state)
        assert result["task_id"] == "#456"
        assert result["task_raw"] == {}

    @pytest.mark.asyncio
    async def test_routes_free_text(self, base_state):
        state = {**base_state, "task_id": "Add OAuth support to the auth module"}
        result = await input_router_node(state)
        assert result["task_id"] == ""
        assert result["task_raw"]["free_text"] == "Add OAuth support to the auth module"

    @pytest.mark.asyncio
    async def test_handles_empty_input(self, base_state):
        state = {**base_state, "task_id": ""}
        result = await input_router_node(state)
        assert result["task_id"] == ""
        assert "error" in result["task_raw"]


class TestReader:
    @pytest.mark.asyncio
    async def test_reads_task(self, base_state, mock_task_provider):
        state = {**base_state, "task_id": "TEST-1"}
        result = await reader_node(state, task_provider=mock_task_provider)
        assert result["task_raw"]["id"] == "TEST-1"
        assert result["task_raw"]["title"] == "Test task"
        mock_task_provider.get_task.assert_called_once_with("TEST-1")

    @pytest.mark.asyncio
    async def test_skips_without_task_id(self, base_state, mock_task_provider):
        state = {**base_state, "task_id": ""}
        await reader_node(state, task_provider=mock_task_provider)
        mock_task_provider.get_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_provider_error(self, base_state, mock_task_provider):
        mock_task_provider.get_task.side_effect = Exception("connection failed")
        state = {**base_state, "task_id": "TEST-1"}
        with pytest.raises(PermanentError, match="Failed to read task"):
            await reader_node(state, task_provider=mock_task_provider)


class TestEnricher:
    @pytest.mark.asyncio
    async def test_enriches_with_no_results(self, base_state, mock_code_provider, mock_doc_provider):
        state = {**base_state, "task_raw": {"title": "test"}}
        result = await enricher_node(state, code_provider=mock_code_provider, doc_provider=mock_doc_provider)
        assert result["has_code_context"] is False
        assert result["has_doc_context"] is False

    @pytest.mark.asyncio
    async def test_enriches_with_code_results(self, base_state, mock_code_provider, mock_doc_provider):
        mock_code_provider.search_code.return_value = [{"content": "def foo(): pass", "source": "code"}]
        state = {**base_state, "task_raw": {"title": "test"}}
        result = await enricher_node(state, code_provider=mock_code_provider, doc_provider=mock_doc_provider)
        assert result["has_code_context"] is True
        assert len(result["enriched_context"]["code_snippets"]) == 1

    @pytest.mark.asyncio
    async def test_degrades_on_code_failure(self, base_state, mock_code_provider, mock_doc_provider):
        mock_code_provider.search_code.side_effect = Exception("MCP down")
        state = {**base_state, "task_raw": {"title": "test"}}
        result = await enricher_node(state, code_provider=mock_code_provider, doc_provider=mock_doc_provider)
        assert result["has_code_context"] is False
        assert len(result["enrichment_warnings"]) > 0


class TestRanker:
    @pytest.mark.asyncio
    async def test_ranks_empty_context(self, base_state):
        result = await ranker_node(base_state)
        assert result["ranked_context"] == []

    @pytest.mark.asyncio
    async def test_filters_by_threshold(self, base_state):
        state = {
            **base_state,
            "enriched_context": {
                "code": [
                    {"content": "relevant", "score": 0.2},
                    {"content": "irrelevant", "score": 0.8},
                ],
            },
        }
        result = await ranker_node(state)
        assert len(result["ranked_context"]) == 1
        assert result["ranked_context"][0]["content"] == "relevant"


class TestStubNodes:
    @pytest.mark.asyncio
    async def test_explainer_stub(self, base_state):
        result = await explainer_node(base_state)
        assert result["plan"] is not None

    @pytest.mark.asyncio
    async def test_executor_stub(self, base_state):
        result = await executor_node(base_state)
        assert "code_changes" in result

    @pytest.mark.asyncio
    async def test_reviewer_stub(self, base_state):
        result = await reviewer_node(base_state)
        assert result["review_result"]["approved"] is True

    @pytest.mark.asyncio
    async def test_doc_writer_stub(self, base_state):
        result = await doc_writer_node(base_state)
        assert result is not None
