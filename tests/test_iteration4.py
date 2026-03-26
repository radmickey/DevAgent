"""Tests for Iteration 4: observability, meta-agent, doc_writer, bench, vector lang filter."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.memory.agent_docs import AgentDocs
from agent.observability import get_tracing_status, setup_tracing
from agent.pipeline.meta_agent import (
    CONFIDENCE_THRESHOLD,
    PromptUpdate,
    _analyze_outcome,
    maybe_update_prompts,
)
from agent.pipeline.nodes.doc_writer import (
    _build_details,
    _build_summary,
    _detect_languages,
    doc_writer_node,
)
from agent.pipeline.state import PipelineState
from tests.bench import BenchmarkReport, TaskResult, run_single_task


# --- Task 4.1: Observability ---

class TestObservability:
    def test_setup_tracing_no_keys(self):
        with patch.dict("os.environ", {}, clear=False):
            with patch.dict("os.environ", {"LANGSMITH_API_KEY": "", "LOGFIRE_TOKEN": ""}, clear=False):
                result = setup_tracing()
                assert result["langsmith"] is False
                assert result["logfire"] is False

    def test_get_tracing_status(self):
        status = get_tracing_status()
        assert isinstance(status, dict)
        assert "langsmith" in status
        assert "logfire" in status

    @patch.dict("os.environ", {"LANGSMITH_API_KEY": "test-key-123"})
    def test_langsmith_setup_with_key(self):
        from agent.observability import _setup_langsmith
        result = _setup_langsmith()
        assert result is True


# --- Task 4.2: Meta-Agent ---

class TestMetaAgent:
    @pytest.mark.asyncio
    async def test_no_feedback_no_update(self):
        result = await maybe_update_prompts("T-1", "", None)
        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_many_iterations(self):
        outcome = {"iteration_count": 4, "review_result": {"findings": []}}
        updates = _analyze_outcome("T-1", "", outcome)
        assert len(updates) >= 1
        assert any(u.node == "executor" for u in updates)

    @pytest.mark.asyncio
    async def test_analyze_rejected_plan(self):
        outcome = {"plan_approved": False}
        updates = _analyze_outcome("T-1", "Plan is too vague", outcome)
        assert any(u.node == "explainer" for u in updates)

    @pytest.mark.asyncio
    async def test_analyze_test_failure(self):
        outcome = {"review_result": {"tests_passed": False, "findings": []}}
        updates = _analyze_outcome("T-1", "", outcome)
        assert any(u.reason == "Tests failed during review" for u in updates)

    @pytest.mark.asyncio
    async def test_analyze_many_error_findings(self):
        findings = [
            {"severity": "error", "message": f"Error {i}"} for i in range(5)
        ]
        outcome = {"review_result": {"findings": findings}}
        updates = _analyze_outcome("T-1", "", outcome)
        assert any("error findings" in u.reason.lower() for u in updates)

    @pytest.mark.asyncio
    async def test_applies_with_longterm(self, tmp_path):
        from agent.memory.longterm import LongtermMemory
        lt = LongtermMemory(db_path=tmp_path / "lt.db")

        outcome = {"iteration_count": 5, "review_result": {"findings": []}}
        result = await maybe_update_prompts("T-1", "", outcome, longterm=lt)

        # The update for iteration_count=5 has confidence 0.5+5*0.1=1.0 > threshold
        assert len(result) >= 1

        history = lt.get_prompt_history("executor")
        assert len(history) >= 1

        patterns = lt.get_patterns(domain="meta_agent")
        assert len(patterns) >= 1
        lt.close()

    @pytest.mark.asyncio
    async def test_skips_low_confidence(self):
        outcome = {"plan_approved": False}
        # Plan rejected feedback has confidence 0.6 < 0.7 threshold
        updates = _analyze_outcome("T-1", "feedback", outcome)
        low_conf = [u for u in updates if u.confidence < CONFIDENCE_THRESHOLD]
        assert len(low_conf) > 0

    def test_prompt_update_model(self):
        update = PromptUpdate(
            node="explainer",
            new_prompt="test prompt",
            reason="test reason",
            confidence=0.8,
        )
        assert update.node == "explainer"
        assert update.confidence == 0.8


# --- Task 4.3: DocWriter ---

class TestDocWriter:
    @pytest.mark.asyncio
    async def test_doc_writer_writes_docs(self, tmp_path):
        docs = AgentDocs(path=tmp_path / "docs.jsonl")
        state: PipelineState = {
            "task_id": "T-1",
            "task_raw": {"title": "Add feature"},
            "plan": {"approach": "test", "steps": []},
            "code_changes": [{"path": "app.py", "action": "modify"}],
            "review_result": {"approved": True, "findings": []},
        }
        result = await doc_writer_node(state, docs=docs)
        assert result is not None

        entries = docs.read_all()
        assert len(entries) == 1
        assert entries[0]["task_id"] == "T-1"

    @pytest.mark.asyncio
    async def test_doc_writer_no_task_id(self):
        state: PipelineState = {"task_id": ""}
        result = await doc_writer_node(state)
        assert result is not None

    def test_build_summary(self):
        summary = _build_summary(
            {"title": "Fix bug"},
            {"approach": "patch the handler"},
            [{"path": "a.py", "action": "modify"}],
            {"approved": True},
        )
        assert "Fix bug" in summary
        assert "1" in summary

    def test_build_details(self):
        details = _build_details(
            {"steps": [{"description": "do it"}]},
            [{"path": "a.py", "action": "create"}],
            {"approved": True, "findings": [{"severity": "info"}]},
        )
        assert len(details["files_changed"]) == 1
        assert details["review"]["approved"] is True
        assert details["review"]["findings_count"] == 1

    def test_detect_languages(self):
        changes = [
            {"path": "app.py", "action": "modify"},
            {"path": "index.ts", "action": "create"},
            {"path": "README.md", "action": "modify"},
        ]
        langs = _detect_languages(changes)
        assert "python" in langs
        assert "typescript" in langs
        assert len(langs) == 2

    def test_detect_languages_empty(self):
        assert _detect_languages([]) == []
        assert _detect_languages(None) == []


# --- Task 4.3: ChromaDB lang filter ---

class TestVectorLangFilter:
    def test_store_with_lang_metadata(self, tmp_path):
        from agent.memory.vector import VectorMemory
        vm = VectorMemory(persist_dir=str(tmp_path / "chroma"))
        vm.store("T-1", [
            {"content": "def hello(): pass", "source": "code", "lang": "python"},
            {"content": "function hello() {}", "source": "code", "lang": "javascript"},
        ])
        # Should store 2 items
        assert vm._collection.count() == 2

    def test_query_with_lang_filter(self, tmp_path):
        from agent.memory.vector import VectorMemory
        vm = VectorMemory(persist_dir=str(tmp_path / "chroma"))
        vm.store("T-1", [
            {"content": "Python function implementation", "source": "code", "lang": "python"},
            {"content": "JavaScript function implementation", "source": "code", "lang": "javascript"},
        ])
        results = vm.query("T-1", "function implementation", lang="python")
        assert len(results) >= 1

    def test_query_empty_collection(self, tmp_path):
        from agent.memory.vector import VectorMemory
        vm = VectorMemory(persist_dir=str(tmp_path / "chroma"))
        results = vm.query("T-1", "anything")
        assert results == []


# --- Task 4.4: pyproject.toml ---

class TestPyprojectToml:
    def test_pyproject_has_scripts(self):
        import tomllib
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["project"]["scripts"]["devagent"] == "agent.interface.cli:main"

    def test_pyproject_has_dependencies(self):
        import tomllib
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        deps = data["project"]["dependencies"]
        assert any("langgraph" in d for d in deps)
        assert any("pydantic-ai" in d for d in deps)
        assert any("chromadb" in d for d in deps)

    def test_pyproject_has_optional_deps(self):
        import tomllib
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        opt = data["project"]["optional-dependencies"]
        assert "dev" in opt
        assert "observability" in opt


# --- Task 4.5: Benchmark ---

class TestBenchmark:
    @pytest.mark.asyncio
    async def test_single_task(self):
        task = {"id": "BENCH-1", "title": "Add OAuth2", "type": "feature"}
        result = await run_single_task(task)
        assert isinstance(result, TaskResult)
        assert result.task_id == "BENCH-1"
        assert result.elapsed_ms > 0

    def test_benchmark_report(self):
        report = BenchmarkReport(
            total_tasks=5,
            successful=4,
            failed=1,
            avg_elapsed_ms=100.0,
            avg_plan_steps=3.5,
        )
        assert report.successful == 4
        assert report.failed == 1

    def test_synthetic_tasks_count(self):
        from tests.bench import SYNTHETIC_TASKS
        assert len(SYNTHETIC_TASKS) == 20
        assert all("id" in t and "title" in t and "type" in t for t in SYNTHETIC_TASKS)
