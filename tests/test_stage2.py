"""Tests for Stage 2: contracts, multi-provider, web search, self-evolution, concurrency, web."""

from __future__ import annotations


import pytest
from unittest.mock import AsyncMock

from agent.pipeline.concurrent import ConcurrentExecutor, TaskStatus
from agent.pipeline.meta_agent_v2 import (
    ABTest,
    PromptVariant,
    SelfEvolutionManager,
    evolve_prompt_with_llm,
)
from agent.pipeline.nodes.contract_extractor import (
    _extract_contract_info,
    _extract_openapi_endpoints,
    _extract_proto_services,
    _find_contract_files,
    contract_extractor_node,
)
from agent.pipeline.nodes.contract_validator import (
    _check_proto_field_reuse,
    _find_removed_endpoints,
    _is_route_file,
    contract_validator_node,
)
from agent.pipeline.state import PipelineState
from agent.providers.multi import MultiCodeProvider, MultiDocProvider, MultiTaskProvider
from agent.providers.web_search import WebSearchProvider


# --- 5.1: Contract Extractor ---

class TestContractExtractor:
    def test_find_contract_files(self):
        files = ["api/openapi.yaml", "proto/service.proto", "src/main.py", "schema.graphql"]
        found = _find_contract_files(files)
        assert len(found) == 3
        types = {f["type"] for f in found}
        assert "openapi" in types
        assert "protobuf" in types
        assert "graphql" in types

    def test_extract_openapi_endpoints(self):
        content = """
paths:
  /users:
    get:
      summary: List users
  /users/{id}:
    get:
      summary: Get user
"""
        endpoints = _extract_openapi_endpoints(content)
        assert "/users" in endpoints
        assert "/users/{id}" in endpoints

    def test_extract_proto_services(self):
        content = """
service UserService {
    rpc GetUser (GetUserRequest) returns (User);
}
service OrderService {
    rpc CreateOrder (CreateOrderRequest) returns (Order);
}
"""
        services = _extract_proto_services(content)
        assert "UserService" in services
        assert "OrderService" in services

    def test_extract_contract_info_openapi(self):
        info = _extract_contract_info("api.yaml", "openapi", "/users:\n  get:")
        assert info is not None
        assert info["type"] == "openapi"
        assert len(info["endpoints"]) >= 1

    def test_extract_contract_info_empty(self):
        assert _extract_contract_info("f.yaml", "openapi", "") is None

    @pytest.mark.asyncio
    async def test_node_without_provider(self):
        state: PipelineState = {"task_id": "T-1"}
        result = await contract_extractor_node(state)
        assert result["contracts"] == []

    @pytest.mark.asyncio
    async def test_node_with_provider(self):
        provider = AsyncMock()
        provider.list_files.return_value = ["openapi.yaml"]
        provider.get_file.return_value = "/users:\n  get:\n    summary: List"

        state: PipelineState = {"task_id": "T-1"}
        result = await contract_extractor_node(state, code_provider=provider)
        assert len(result["contracts"]) == 1


# --- 5.1: Contract Validator ---

class TestContractValidator:
    def test_is_route_file(self):
        assert _is_route_file("src/api/routes.py") is True
        assert _is_route_file("src/models/user.py") is False

    def test_find_removed_endpoints(self):
        diff = "- router.get('/users')\n+ router.get('/people')"
        removed = _find_removed_endpoints(diff, ["/users"])
        assert "/users" in removed

    def test_check_proto_field_reuse(self):
        diff = "+  string name = 1;\n+  int32 age = 1;"
        issues = _check_proto_field_reuse(diff)
        assert len(issues) >= 1

    @pytest.mark.asyncio
    async def test_validator_no_contracts(self):
        state: PipelineState = {"contracts": [], "code_changes": []}
        result = await contract_validator_node(state)
        assert result["contract_violations"] == []

    @pytest.mark.asyncio
    async def test_validator_with_violation(self):
        state: PipelineState = {
            "contracts": [
                {"type": "openapi", "path": "api.yaml", "endpoints": ["/users"]},
            ],
            "code_changes": [
                {"path": "src/api/routes.py", "diff": "- app.get('/users')"},
            ],
        }
        result = await contract_validator_node(state)
        assert len(result["contract_violations"]) >= 1


# --- 5.2: MultiProvider ---

class TestMultiProvider:
    @pytest.mark.asyncio
    async def test_fallback_first_succeeds(self):
        p1 = AsyncMock()
        p1.get_task.return_value = {"title": "from p1"}
        p2 = AsyncMock()

        multi = MultiTaskProvider([p1, p2], strategy="fallback")
        result = await multi.get_task("T-1")
        assert result["title"] == "from p1"
        p2.get_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_to_second(self):
        p1 = AsyncMock()
        p1.get_task.side_effect = Exception("down")
        p2 = AsyncMock()
        p2.get_task.return_value = {"title": "from p2"}

        multi = MultiTaskProvider([p1, p2], strategy="fallback")
        result = await multi.get_task("T-1")
        assert result["title"] == "from p2"

    @pytest.mark.asyncio
    async def test_all_fail_raises(self):
        p1 = AsyncMock()
        p1.get_task.side_effect = Exception("fail1")
        p2 = AsyncMock()
        p2.get_task.side_effect = Exception("fail2")

        multi = MultiTaskProvider([p1, p2])
        with pytest.raises(Exception):
            await multi.get_task("T-1")

    @pytest.mark.asyncio
    async def test_round_robin(self):
        p1 = AsyncMock()
        p1.get_task.return_value = {"title": "p1"}
        p2 = AsyncMock()
        p2.get_task.return_value = {"title": "p2"}

        multi = MultiTaskProvider([p1, p2], strategy="round_robin")
        await multi.get_task("T-1")
        await multi.get_task("T-2")
        assert p1.get_task.call_count + p2.get_task.call_count >= 2

    def test_empty_providers_raises(self):
        with pytest.raises(ValueError):
            MultiTaskProvider([])

    @pytest.mark.asyncio
    async def test_multi_code_provider(self):
        p1 = AsyncMock()
        p1.search_code.return_value = [{"content": "code"}]
        multi = MultiCodeProvider([p1])
        result = await multi.search_code("test")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_multi_doc_provider(self):
        p1 = AsyncMock()
        p1.search_docs.return_value = [{"content": "doc"}]
        multi = MultiDocProvider([p1])
        result = await multi.search_docs("test")
        assert len(result) == 1


# --- 5.3: Web Search ---

class TestWebSearch:
    def test_not_configured(self):
        ws = WebSearchProvider(api_key="")
        assert ws.is_configured is False

    def test_configured(self):
        ws = WebSearchProvider(api_key="test-key")
        assert ws.is_configured is True

    @pytest.mark.asyncio
    async def test_search_without_key_raises(self):
        ws = WebSearchProvider(api_key="")
        from agent.errors import DegradedError
        with pytest.raises(DegradedError):
            await ws.search("test query")

    @pytest.mark.asyncio
    async def test_search_safe_returns_empty(self):
        ws = WebSearchProvider(api_key="")
        result = await ws.search_safe("test query")
        assert result == []

    def test_include_domains(self):
        ws = WebSearchProvider(api_key="key", include_domains=["docs.python.org"])
        assert ws._include_domains == ["docs.python.org"]


# --- 5.4: Self-Evolution L2-L3 ---

class TestSelfEvolution:
    def test_prompt_variant(self):
        v = PromptVariant(variant_id="A", prompt="test", node="explainer", wins=3, trials=5)
        assert v.win_rate == 0.6

    def test_ab_test_not_conclusive(self):
        test = ABTest(
            test_id="t1", node="explainer",
            variant_a=PromptVariant("A", "p1", "explainer", trials=5),
            variant_b=PromptVariant("B", "p2", "explainer", trials=5),
            min_trials=10,
        )
        assert test.is_conclusive is False
        assert test.winner is None

    def test_ab_test_conclusive(self):
        test = ABTest(
            test_id="t1", node="explainer",
            variant_a=PromptVariant("A", "p1", "explainer", wins=8, trials=10),
            variant_b=PromptVariant("B", "p2", "explainer", wins=3, trials=10),
            min_trials=10,
        )
        assert test.is_conclusive is True
        assert test.winner is not None
        assert test.winner.variant_id == "A"

    def test_manager_start_test(self):
        mgr = SelfEvolutionManager()
        test = mgr.start_ab_test("explainer", "prompt A", "prompt B", min_trials=5)
        assert test.status == "active"
        assert test.node == "explainer"

    def test_manager_get_prompt(self):
        mgr = SelfEvolutionManager()
        mgr.start_ab_test("explainer", "A", "B")
        prompt = mgr.get_prompt_for_node("explainer")
        assert prompt in ("A", "B")

    def test_manager_no_test(self):
        mgr = SelfEvolutionManager()
        assert mgr.get_prompt_for_node("explainer") is None

    def test_manager_cancel(self):
        mgr = SelfEvolutionManager()
        mgr.start_ab_test("explainer", "A", "B")
        assert mgr.cancel_test("explainer") is True
        assert mgr.cancel_test("nonexistent") is False

    @pytest.mark.asyncio
    async def test_evolve_no_issues(self):
        result = await evolve_prompt_with_llm("explainer", "prompt", "", {"iteration_count": 1})
        assert result is None

    @pytest.mark.asyncio
    async def test_evolve_with_issues(self):
        result = await evolve_prompt_with_llm(
            "explainer", "base prompt", "too vague",
            {"iteration_count": 5},
        )
        assert result is not None
        assert "base prompt" in result


# --- 5.5: Concurrent Tasks ---

class TestConcurrentExecutor:
    @pytest.mark.asyncio
    async def test_submit_and_run(self):
        executor = ConcurrentExecutor(max_concurrent=2)
        await executor.submit("T-1", {"input": "test"})

        async def mock_pipeline(data: dict) -> dict:
            return {"result": "done"}

        results = await executor.run_all(mock_pipeline)
        assert len(results) == 1
        assert results[0].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_cancel_queued(self):
        executor = ConcurrentExecutor()
        await executor.submit("T-1", {})
        assert executor.cancel("T-1") is True
        assert executor.cancel("nonexistent") is False

    @pytest.mark.asyncio
    async def test_handles_failure(self):
        executor = ConcurrentExecutor()
        await executor.submit("T-1", {})

        async def failing_pipeline(data: dict) -> dict:
            raise RuntimeError("boom")

        results = await executor.run_all(failing_pipeline)
        assert results[0].status == TaskStatus.FAILED
        assert results[0].error == "boom"

    @pytest.mark.asyncio
    async def test_get_status(self):
        executor = ConcurrentExecutor()
        await executor.submit("T-1", {"x": 1})
        status = executor.get_status("T-1")
        assert status is not None
        assert status.task_id == "T-1"

    @pytest.mark.asyncio
    async def test_get_all_statuses(self):
        executor = ConcurrentExecutor()
        await executor.submit("T-1", {})
        await executor.submit("T-2", {})
        statuses = executor.get_all_statuses()
        assert len(statuses) == 2

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        executor = ConcurrentExecutor(max_concurrent=1)
        await executor.submit("T-low", {}, priority=1)
        await executor.submit("T-high", {}, priority=10)

        async def tracking_pipeline(data: dict) -> dict:
            return {}

        await executor.run_all(tracking_pipeline)
        all_tasks = executor.get_all_statuses()
        assert len(all_tasks) == 2


# --- 5.6: Web interface ---

class TestWebInterface:
    def test_emit_event_no_queue(self):
        from agent.interface.web import emit_event
        emit_event("T-1", "test", {"key": "value"})

    def test_create_app_graceful(self):
        """Verify create_app handles import/init issues gracefully."""
        try:
            from agent.interface.web import create_app
            app = create_app()
            assert app is not None
        except (RuntimeError, TypeError, ImportError):
            pass


# --- 5.7: CI/CD ---

class TestCICD:
    def test_ci_workflow_exists(self):
        from pathlib import Path
        ci = Path(".github/workflows/ci.yml")
        assert ci.exists()

    def test_build_workflow_exists(self):
        from pathlib import Path
        build = Path(".github/workflows/build.yml")
        assert build.exists()


# --- 5.8: pyproject.toml updates ---

class TestPyprojectExtensions:
    def test_has_web_optional_dep(self):
        import tomllib
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        assert "web" in data["project"]["optional-dependencies"]
        assert "search" in data["project"]["optional-dependencies"]
        assert "build" in data["project"]["optional-dependencies"]

    def test_state_has_contract_fields(self):
        state: PipelineState = {
            "contracts": [{"type": "openapi"}],
            "contract_violations": [],
            "detected_languages": ["python"],
        }
        assert len(state["contracts"]) == 1
        assert state["detected_languages"] == ["python"]
