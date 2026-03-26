"""Tests for Iteration 3: models, explainer, executor, reviewer, sanitizer, dry-run."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from agent.pipeline.models import (
    ExplainerResult, ExecutorResult, FileChange, NodeDeps,
    PlanStep, ReviewFinding, ReviewResult, TaskInfo,
)
from agent.pipeline.nodes.explainer import explainer_node
from agent.pipeline.nodes.executor import executor_node
from agent.pipeline.nodes.reviewer import reviewer_node
from agent.pipeline.prompts.explainer import build_explainer_prompt
from agent.pipeline.prompts.executor import build_executor_prompt
from agent.pipeline.prompts.reviewer import build_reviewer_prompt
from agent.pipeline.state import PipelineState
from agent.security.sanitizer import sanitize, sanitize_if_needed, should_sanitize


@pytest.fixture
def base_state() -> PipelineState:
    return PipelineState(
        task_id="TEST-1",
        task_raw={"title": "Add OAuth support", "description": "Implement OAuth2"},
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


# --- Task 3.1: Pydantic Models ---

class TestPydanticModels:
    def test_task_info(self):
        task = TaskInfo(
            goal="Add OAuth",
            type="feature",
            domain=["auth"],
            constraints=[],
            open_questions=[],
            priority_comments=[],
        )
        assert task.goal == "Add OAuth"
        assert task.type == "feature"

    def test_explainer_result(self):
        result = ExplainerResult(
            summary="Add OAuth2 flow",
            approach="Extend auth module",
            steps=[PlanStep(description="Create OAuth handler", action="create", file_path="auth/oauth.py")],
            risks=["Token expiry handling"],
            estimated_complexity="medium",
        )
        assert len(result.steps) == 1
        assert result.steps[0].file_path == "auth/oauth.py"

    def test_executor_result(self):
        result = ExecutorResult(
            files_changed=[FileChange(path="auth/oauth.py", action="create", content="code")],
            commands_run=["git checkout -b feat/oauth"],
            side_effects=["branch_created"],
        )
        assert len(result.files_changed) == 1

    def test_review_result(self):
        result = ReviewResult(
            approved=False,
            findings=[ReviewFinding(severity="error", message="Missing error handling", file_path="oauth.py")],
            tests_passed=True,
            static_analysis_passed=True,
            summary="Needs error handling",
        )
        assert not result.approved
        assert len(result.findings) == 1

    def test_node_deps(self):
        deps = NodeDeps(task_id="T-1")
        assert deps.task_id == "T-1"
        assert deps.task_provider is None
        assert deps.context == {}

    def test_model_serialization(self):
        result = ExplainerResult(summary="test", approach="test")
        d = result.model_dump()
        assert isinstance(d, dict)
        assert d["summary"] == "test"
        restored = ExplainerResult.model_validate(d)
        assert restored.summary == "test"


# --- Task 3.2/3.3: Explainer with Pydantic AI ---

class TestExplainerNode:
    @pytest.mark.asyncio
    async def test_explainer_handles_failure_gracefully(self, base_state):
        """Agent with 'test' model will fail — verify graceful fallback."""
        result = await explainer_node(base_state)
        assert "plan" in result
        plan = result["plan"]
        assert isinstance(plan, dict)
        assert "summary" in plan

    def test_build_explainer_prompt_basic(self, base_state):
        prompt = build_explainer_prompt(base_state)
        assert "Add OAuth support" in prompt
        assert "No code context" in prompt
        assert "No documentation context" in prompt

    def test_build_explainer_prompt_with_context(self, base_state):
        state = {
            **base_state,
            "has_code_context": True,
            "has_doc_context": True,
            "has_similar_tasks": True,
            "ranked_context": [{"content": "def login():", "source": "code"}],
        }
        prompt = build_explainer_prompt(state)
        assert "No code context" not in prompt
        assert "Similar historical tasks" in prompt

    def test_build_explainer_prompt_with_warnings(self, base_state):
        state = {**base_state, "enrichment_warnings": ["MCP timeout"]}
        prompt = build_explainer_prompt(state)
        assert "MCP timeout" in prompt


# --- Task 3.4: Executor ---

class TestExecutorNode:
    @pytest.mark.asyncio
    async def test_executor_no_plan(self, base_state):
        state = {**base_state, "plan": {}}
        result = await executor_node(state)
        assert result["code_changes"] == []

    @pytest.mark.asyncio
    async def test_executor_no_steps(self, base_state):
        state = {**base_state, "plan": {"summary": "test", "steps": []}}
        result = await executor_node(state)
        assert result["code_changes"] == []

    @pytest.mark.asyncio
    async def test_executor_handles_failure(self, base_state):
        state = {
            **base_state,
            "plan": {"summary": "test", "approach": "test", "steps": [{"description": "do"}]},
        }
        result = await executor_node(state)
        assert "code_changes" in result
        assert "iteration_count" in result

    def test_build_executor_prompt(self):
        plan = {
            "approach": "Extend auth",
            "steps": [
                {"description": "Create handler", "file_path": "auth.py", "action": "create"},
            ],
        }
        prompt = build_executor_prompt(plan, {"title": "Add OAuth"})
        assert "Add OAuth" in prompt
        assert "Create handler" in prompt


# --- Task 3.5: Side Effects in Executor ---

class TestExecutorSideEffects:
    @pytest.mark.asyncio
    async def test_effects_tracker_integration(self, base_state, tmp_path):
        from agent.memory.effects import SideEffectTracker
        tracker = SideEffectTracker(db_path=tmp_path / "fx.db")
        deps = NodeDeps(task_id="TEST-1", effects_tracker=tracker)

        state = {
            **base_state,
            "plan": {"summary": "t", "approach": "t", "steps": [{"description": "x"}]},
        }
        await executor_node(state, deps=deps)
        tracker.close()


# --- Task 3.6: Reviewer v2 ---

class TestReviewerNode:
    @pytest.mark.asyncio
    async def test_reviewer_no_changes(self, base_state):
        state = {**base_state, "code_changes": []}
        result = await reviewer_node(state)
        review = result["review_result"]
        assert review["approved"] is True
        assert "No code changes" in review["summary"]

    @pytest.mark.asyncio
    async def test_reviewer_with_changes_handles_failure(self, base_state):
        state = {
            **base_state,
            "code_changes": [{"path": "test.py", "action": "modify", "diff": "+pass"}],
            "plan": {"summary": "test plan"},
        }
        result = await reviewer_node(state)
        assert "review_result" in result
        review = result["review_result"]
        assert isinstance(review, dict)
        assert "approved" in review

    def test_build_reviewer_prompt(self):
        changes = [{"path": "auth.py", "action": "modify", "diff": "+def login():"}]
        plan = {"summary": "Add auth"}
        prompt = build_reviewer_prompt(changes, plan)
        assert "auth.py" in prompt
        assert "Add auth" in prompt

    def test_build_reviewer_prompt_with_tests(self):
        changes = [{"path": "a.py", "action": "create", "content": "pass"}]
        plan = {"summary": "plan"}
        prompt = build_reviewer_prompt(changes, plan, test_output="5 passed", lint_output="ok")
        assert "5 passed" in prompt
        assert "ok" in prompt


# --- Task 3.7: --dry-run ---

class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_flag_in_state(self, base_state):
        state = {**base_state, "dry_run": True}
        assert state["dry_run"] is True

    def test_dry_run_graph_edge(self):
        from agent.pipeline.graph import _check_dry_run
        from langgraph.graph import END

        state_dry: PipelineState = {"dry_run": True}  # type: ignore[typeddict-item]
        assert _check_dry_run(state_dry) == END

        state_normal: PipelineState = {"dry_run": False}  # type: ignore[typeddict-item]
        assert _check_dry_run(state_normal) == "hitl"


# --- Task 3.8: Sanitizer ---

class TestSanitizer:
    def test_sanitize_api_key(self):
        text = "api_key=sk-123456789abcdef"
        result = sanitize(text)
        assert "sk-123456789abcdef" not in result
        assert "REDACTED" in result

    def test_sanitize_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.test"
        result = sanitize(text)
        assert "eyJhbGciOiJIUzI1NiJ9" not in result

    def test_sanitize_email(self):
        text = "Contact user@company.com for help"
        result = sanitize(text)
        assert "user@company.com" not in result
        assert "EMAIL" in result

    def test_sanitize_ip_address(self):
        text = "Connect to 192.168.1.100"
        result = sanitize(text)
        assert "192.168.1.100" not in result
        assert "IP" in result

    def test_sanitize_db_url(self):
        text = "postgresql://user:pass@host:5432/db"
        result = sanitize(text)
        assert "user:pass" not in result
        assert "DB_URL" in result

    def test_sanitize_private_key(self):
        text = "-----BEGIN PRIVATE KEY-----\nMIIEvg==\n-----END PRIVATE KEY-----"
        result = sanitize(text)
        assert "MIIEvg" not in result
        assert "PRIVATE_KEY" in result

    def test_sanitize_preserves_safe_text(self):
        text = "def hello():\n    return 'world'"
        assert sanitize(text) == text

    def test_sanitize_aws_keys(self):
        text = "aws_access_key_id = AKIAIOSFODNN7EXAMPLE"
        result = sanitize(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    @patch("agent.config.get_config")
    def test_should_sanitize_true(self, mock_config):
        mock_config.return_value.use_external_llm = True
        assert should_sanitize() is True

    @patch("agent.config.get_config")
    def test_should_sanitize_false(self, mock_config):
        mock_config.return_value.use_external_llm = False
        assert should_sanitize() is False

    @patch("agent.security.sanitizer.should_sanitize", return_value=False)
    def test_sanitize_if_needed_no_op(self, _):
        text = "api_key=secret123"
        assert sanitize_if_needed(text) == text

    @patch("agent.security.sanitizer.should_sanitize", return_value=True)
    def test_sanitize_if_needed_active(self, _):
        text = "api_key=secret123"
        result = sanitize_if_needed(text)
        assert "secret123" not in result
