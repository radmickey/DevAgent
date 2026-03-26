"""Tests for Iteration 2: enricher v2, sub-agents, ranker v2, dual LLM, longterm patterns."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent.pipeline.state import PipelineState
from agent.pipeline.nodes.enricher import enricher_node
from agent.pipeline.nodes.ranker import ranker_node
from agent.memory.budget import count_tokens, fit_context
from agent.memory.cost import CostTracker
from agent.memory.longterm import LongtermMemory
from agent.llm import get_model_for_node, get_fast_model, get_strong_model
from agent.providers.base import TaskProvider, CodeProvider, DocProvider


@pytest.fixture
def mock_task_provider():
    provider = AsyncMock(spec=TaskProvider)
    provider.get_task.return_value = {
        "id": "TEST-1", "title": "Test task",
        "description": "desc", "comments": [], "labels": [], "status": "open",
    }
    provider.get_comments.return_value = [
        {"author": "dev", "text": "Check the auth module"},
    ]
    return provider


@pytest.fixture
def mock_code_provider():
    provider = AsyncMock(spec=CodeProvider)
    provider.search_code.return_value = [
        {"path": "auth.py", "content": "def login(): pass", "source": "code"},
    ]
    return provider


@pytest.fixture
def mock_doc_provider():
    provider = AsyncMock(spec=DocProvider)
    provider.search_docs.return_value = [
        {"title": "Auth Guide", "content": "Use OAuth2", "source": "doc"},
    ]
    return provider


@pytest.fixture
def base_state() -> PipelineState:
    return PipelineState(
        task_id="TEST-1",
        task_raw={"title": "Add OAuth support", "description": "Add OAuth2 to auth module"},
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


# --- Enricher v2 Tests ---

class TestEnricherV2:
    @pytest.mark.asyncio
    async def test_enriches_all_sources(self, base_state, mock_code_provider, mock_doc_provider, mock_task_provider):
        result = await enricher_node(
            base_state,
            code_provider=mock_code_provider,
            doc_provider=mock_doc_provider,
            task_provider=mock_task_provider,
        )
        assert result["has_code_context"] is True
        assert result["has_doc_context"] is True
        assert "code_snippets" in result["enriched_context"]
        assert "doc_pages" in result["enriched_context"]
        assert "task_comments" in result["enriched_context"]

    @pytest.mark.asyncio
    async def test_gather_return_exceptions(self, base_state, mock_code_provider, mock_doc_provider):
        """gather(return_exceptions=True) must not crash on individual failures."""
        mock_code_provider.search_code.side_effect = ConnectionError("MCP down")
        result = await enricher_node(
            base_state,
            code_provider=mock_code_provider,
            doc_provider=mock_doc_provider,
        )
        assert result["has_code_context"] is False
        assert result["has_doc_context"] is True
        assert len(result["enrichment_warnings"]) == 1
        assert "code_snippets" in result["enrichment_warnings"][0]

    @pytest.mark.asyncio
    async def test_all_sources_fail_gracefully(self, base_state):
        code_prov = AsyncMock(spec=CodeProvider)
        doc_prov = AsyncMock(spec=DocProvider)
        code_prov.search_code.side_effect = TimeoutError("timeout")
        doc_prov.search_docs.side_effect = ConnectionError("down")

        result = await enricher_node(
            base_state, code_provider=code_prov, doc_provider=doc_prov,
        )
        assert result["has_code_context"] is False
        assert result["has_doc_context"] is False
        assert len(result["enrichment_warnings"]) == 2

    @pytest.mark.asyncio
    async def test_has_similar_tasks_flag(self, base_state, mock_code_provider, mock_doc_provider):
        vector_memory = MagicMock()
        vector_memory.query.return_value = [
            {"content": "similar task pattern", "score": 0.2, "source": "vector"},
        ]

        result = await enricher_node(
            base_state,
            code_provider=mock_code_provider,
            doc_provider=mock_doc_provider,
            vector_memory=vector_memory,
        )
        assert result["has_similar_tasks"] is True
        assert "similar_tasks" in result["enriched_context"]

    @pytest.mark.asyncio
    async def test_without_task_provider(self, base_state, mock_code_provider, mock_doc_provider):
        result = await enricher_node(
            base_state, code_provider=mock_code_provider, doc_provider=mock_doc_provider,
        )
        assert "task_comments" not in result["enriched_context"]


# --- Ranker v2 Tests ---

class TestRankerV2:
    @pytest.mark.asyncio
    async def test_filters_by_threshold(self, base_state):
        state = {
            **base_state,
            "enriched_context": {
                "code_snippets": [
                    {"content": "relevant code", "score": 0.1},
                    {"content": "somewhat relevant", "score": 0.3},
                    {"content": "irrelevant", "score": 0.9},
                ],
            },
        }
        result = await ranker_node(state)
        contents = [r["content"] for r in result["ranked_context"]]
        assert "relevant code" in contents
        assert "somewhat relevant" in contents
        assert "irrelevant" not in contents

    @pytest.mark.asyncio
    async def test_fallback_when_nothing_below_threshold(self, base_state):
        state = {
            **base_state,
            "enriched_context": {
                "code_snippets": [
                    {"content": "only option", "score": 0.9},
                ],
            },
        }
        result = await ranker_node(state)
        assert len(result["ranked_context"]) > 0

    @pytest.mark.asyncio
    async def test_empty_context(self, base_state):
        result = await ranker_node(base_state)
        assert result["ranked_context"] == []

    @pytest.mark.asyncio
    async def test_sorted_by_score(self, base_state):
        state = {
            **base_state,
            "enriched_context": {
                "code_snippets": [
                    {"content": "b", "score": 0.2},
                    {"content": "a", "score": 0.1},
                    {"content": "c", "score": 0.3},
                ],
            },
        }
        result = await ranker_node(state)
        scores = [r["score"] for r in result["ranked_context"]]
        assert scores == sorted(scores)


# --- Dual LLM Tests ---

class TestDualLLM:
    def setup_method(self) -> None:
        from agent.config import _invalidate_config_cache
        from agent.llm import invalidate_registry
        _invalidate_config_cache()
        invalidate_registry()
        self._orig_env: dict[str, str | None] = {}
        for key in ("LLM_STRONG_MODEL", "LLM_FAST_MODEL"):
            self._orig_env[key] = os.environ.pop(key, None)
        import agent.config as _cfg
        self._orig_path = _cfg.SETTINGS_PATH
        _cfg.SETTINGS_PATH = Path("/tmp/_devagent_test_no_such_file.yaml")

    def teardown_method(self) -> None:
        import agent.config as _cfg
        _cfg.SETTINGS_PATH = self._orig_path
        for key, val in self._orig_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        from agent.config import _invalidate_config_cache
        from agent.llm import invalidate_registry
        _invalidate_config_cache()
        invalidate_registry()

    def test_fast_model_for_enricher(self):
        model = get_model_for_node("enricher")
        fast = get_fast_model()
        assert model == fast

    def test_strong_model_for_explainer(self):
        model = get_model_for_node("explainer")
        strong = get_strong_model()
        assert model == strong

    def test_strong_model_for_executor(self):
        model = get_model_for_node("executor")
        strong = get_strong_model()
        assert model == strong

    def test_strong_model_for_reviewer(self):
        model = get_model_for_node("reviewer")
        strong = get_strong_model()
        assert model == strong

    def test_fast_model_for_sub_agents(self):
        fast = get_fast_model()
        assert get_model_for_node("code_search") == fast
        assert get_model_for_node("doc_search") == fast
        assert get_model_for_node("task_search") == fast
        assert get_model_for_node("diff_agent") == fast

    def test_unknown_node_defaults_to_fast(self):
        model = get_model_for_node("unknown_node")
        assert model == get_fast_model()


# --- Cost Tracker v2 Tests ---

class TestCostTrackerV2:
    @pytest.fixture
    def tracker(self, tmp_path):
        t = CostTracker(db_path=tmp_path / "cost.db")
        yield t
        t.close()

    def test_summary_by_model(self, tracker):
        tracker.log("T-1", "enricher", "claude-haiku", 100, 50, 200, 0.0001)
        tracker.log("T-1", "explainer", "claude-sonnet", 500, 200, 2000, 0.005)
        tracker.log("T-1", "reviewer", "claude-sonnet", 300, 100, 1500, 0.003)

        summary = tracker.get_summary_by_model()
        assert len(summary) == 2
        sonnet = next(s for s in summary if s["model"] == "claude-sonnet")
        assert sonnet["calls"] == 2
        assert sonnet["cost_usd"] == pytest.approx(0.008)

    def test_summary_by_node(self, tracker):
        tracker.log("T-1", "enricher", "haiku", 100, 50, 200, 0.0001)
        tracker.log("T-1", "explainer", "sonnet", 500, 200, 2000, 0.005)

        summary = tracker.get_summary_by_node()
        assert len(summary) == 2


# --- Longterm Memory v2 Tests ---

class TestLongtermMemoryV2:
    @pytest.fixture
    def memory(self, tmp_path):
        m = LongtermMemory(db_path=tmp_path / "longterm.db")
        yield m
        m.close()

    def test_save_and_get_pattern(self, memory):
        pid = memory.save_pattern(
            domain="auth", pattern_type="common_mistake",
            content="Always handle token expiry", source_task="T-42",
        )
        assert pid > 0
        patterns = memory.get_patterns(domain="auth")
        assert len(patterns) == 1
        assert patterns[0]["content"] == "Always handle token expiry"

    def test_filter_by_domain(self, memory):
        memory.save_pattern("auth", "mistake", "token expiry")
        memory.save_pattern("api", "convention", "REST naming")
        assert len(memory.get_patterns(domain="auth")) == 1
        assert len(memory.get_patterns(domain="api")) == 1

    def test_filter_by_type(self, memory):
        memory.save_pattern("auth", "mistake", "token expiry")
        memory.save_pattern("auth", "convention", "use BaseOAuthProvider")
        assert len(memory.get_patterns(pattern_type="mistake")) == 1

    def test_count_patterns(self, memory):
        memory.save_pattern("auth", "mistake", "a")
        memory.save_pattern("auth", "mistake", "b")
        memory.save_pattern("api", "convention", "c")
        counts = memory.count_patterns()
        assert counts["auth"] == 2
        assert counts["api"] == 1

    def test_pattern_with_metadata(self, memory):
        memory.save_pattern(
            "auth", "review_comment", "Check token refresh",
            metadata={"frequency": 23, "files": ["auth.py"]},
        )
        p = memory.get_patterns(domain="auth")[0]
        assert p["metadata"]["frequency"] == 23

    def test_prompt_history(self, memory):
        memory.save_prompt_version("explainer", "v1", "init")
        memory.save_prompt_version("explainer", "v2", "improved")
        history = memory.get_prompt_history("explainer")
        assert len(history) == 2
        assert history[0]["active"] is True
        assert history[1]["active"] is False


# --- Token Budget Tests (already work, validate integration) ---

class TestBudgetIntegration:
    def test_count_tokens_realistic(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokens = count_tokens(text)
        assert 80 < tokens < 200

    def test_fit_context_respects_budget(self):
        items = [
            {"content": "important " * 100, "relevance": 0.9},
            {"content": "medium " * 100, "relevance": 0.5},
            {"content": "low " * 100, "relevance": 0.1},
        ]
        result = fit_context(items, max_tokens=200)
        assert len(result) <= 3
        assert len(result) > 0


# --- Sub-agent structure tests (import validation, no LLM calls) ---

class TestSubAgentStructure:
    def test_code_search_agent_exists(self):
        from agent.skills.code_search import code_search_agent, CodeSearchResult
        assert code_search_agent is not None
        assert CodeSearchResult is not None

    def test_doc_search_agent_exists(self):
        from agent.skills.doc_search import doc_search_agent
        assert doc_search_agent is not None

    def test_task_search_agent_exists(self):
        from agent.skills.task_search import task_search_agent
        assert task_search_agent is not None

    def test_diff_agent_exists(self):
        from agent.skills.diff_agent import diff_agent
        assert diff_agent is not None
