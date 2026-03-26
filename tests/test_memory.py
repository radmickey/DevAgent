"""Tests for memory modules: effects, budget, cost, longterm, vector, agent_docs."""

from __future__ import annotations


import pytest

from agent.memory.budget import count_tokens, fit_context
from agent.memory.cost import CostTracker
from agent.memory.effects import SideEffectTracker
from agent.memory.longterm import LongtermMemory
from agent.memory.agent_docs import AgentDocs


class TestSideEffectTracker:
    @pytest.fixture
    def tracker(self, tmp_path):
        t = SideEffectTracker(db_path=tmp_path / "effects.db")
        yield t
        t.close()

    def test_record_and_get(self, tracker):
        tracker.record("T-1", "branch_created", {"branch": "feat-1", "repo": "main"})
        effects = tracker.get("T-1")
        assert len(effects) == 1
        assert effects[0]["effect_type"] == "branch_created"
        assert effects[0]["details"]["branch"] == "feat-1"

    def test_rollback_plan(self, tracker):
        tracker.record("T-1", "branch_created", {"branch": "feat-1"})
        tracker.record("T-1", "file_written", {"path": "/tmp/test.py"})
        plan = tracker.get_rollback_plan("T-1")
        assert len(plan) == 2
        assert plan[0]["action"] == "delete_file"
        assert plan[1]["action"] == "delete_branch"

    def test_mark_interrupted(self, tracker):
        tracker.record("T-1", "branch_created", {"branch": "feat-1"})
        tracker.mark_interrupted("T-1")
        effects = tracker.get("T-1")
        assert effects[0]["interrupted"] is True

    def test_empty_rollback(self, tracker):
        plan = tracker.get_rollback_plan("NONEXISTENT")
        assert plan == []


class TestTokenBudget:
    def test_count_tokens_basic(self):
        tokens = count_tokens("Hello, world!")
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_count_tokens_empty(self):
        assert count_tokens("") == 0

    def test_fit_context_within_budget(self):
        items = [
            {"content": "short text", "relevance": 0.9},
            {"content": "another text", "relevance": 0.5},
        ]
        result = fit_context(items, max_tokens=1000)
        assert len(result) == 2

    def test_fit_context_overflow(self):
        items = [
            {"content": "x " * 5000, "relevance": 0.9},
            {"content": "small", "relevance": 0.5},
        ]
        result = fit_context(items, max_tokens=100)
        assert len(result) < 2


class TestCostTracker:
    @pytest.fixture
    def tracker(self, tmp_path):
        t = CostTracker(db_path=tmp_path / "cost.db")
        yield t
        t.close()

    def test_log_and_get(self, tracker):
        tracker.log(
            task_id="T-1", node="explainer", model="claude-sonnet",
            tokens_in=100, tokens_out=50, latency_ms=500, cost_usd=0.001,
        )
        cost = tracker.get_task_cost("T-1")
        assert cost == pytest.approx(0.001)

    def test_total_cost(self, tracker):
        tracker.log("T-1", "explainer", "claude", 100, 50, 500, 0.001)
        tracker.log("T-2", "executor", "claude", 200, 100, 1000, 0.003)
        assert tracker.get_total_cost() == pytest.approx(0.004)

    def test_empty_cost(self, tracker):
        assert tracker.get_task_cost("NONE") == 0.0


class TestLongtermMemory:
    @pytest.fixture
    def memory(self, tmp_path):
        m = LongtermMemory(db_path=tmp_path / "longterm.db")
        yield m
        m.close()

    def test_save_and_get_prompt(self, memory):
        hash_val = memory.save_prompt_version("explainer", "You are an expert", "initial")
        assert len(hash_val) == 8
        prompt = memory.get_active_prompt("explainer")
        assert prompt == "You are an expert"

    def test_version_replaces_previous(self, memory):
        memory.save_prompt_version("explainer", "version 1", "init")
        memory.save_prompt_version("explainer", "version 2", "update")
        assert memory.get_active_prompt("explainer") == "version 2"

    def test_rollback_prompt(self, memory):
        h1 = memory.save_prompt_version("explainer", "version 1", "init")
        memory.save_prompt_version("explainer", "version 2", "update")
        success = memory.rollback_prompt("explainer", h1)
        assert success
        assert memory.get_active_prompt("explainer") == "version 1"

    def test_rollback_nonexistent_hash(self, memory):
        assert memory.rollback_prompt("explainer", "00000000") is False

    def test_get_prompt_nonexistent_node(self, memory):
        assert memory.get_active_prompt("nonexistent") is None


class TestAgentDocs:
    def test_write_and_read(self, tmp_path):
        docs = AgentDocs(path=tmp_path / "docs.jsonl")
        docs.write("T-1", "Implemented feature X", {"files": ["a.py"]})
        docs.write("T-2", "Fixed bug Y", {"files": ["b.py"]})
        entries = docs.read_all()
        assert len(entries) == 2
        assert entries[0]["task_id"] == "T-1"

    def test_read_empty(self, tmp_path):
        docs = AgentDocs(path=tmp_path / "nonexistent.jsonl")
        assert docs.read_all() == []
