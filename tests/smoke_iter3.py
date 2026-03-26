"""Smoke test for Iteration 3: models, agents, prompts, sanitizer."""

from __future__ import annotations



def main():
    print("[1/6] Importing models...")
    from agent.pipeline.models import (
        ExplainerResult, NodeDeps,
        PlanStep,
    )
    assert NodeDeps(task_id="T-1").task_id == "T-1"

    print("[2/6] Importing agents...")
    from agent.pipeline.nodes.explainer import explainer_agent
    from agent.pipeline.nodes.executor import executor_agent
    from agent.pipeline.nodes.reviewer import reviewer_agent
    assert explainer_agent is not None
    assert executor_agent is not None
    assert reviewer_agent is not None

    print("[3/6] Testing prompts...")
    from agent.pipeline.prompts.explainer import build_explainer_prompt
    prompt = build_explainer_prompt({"task_raw": {"title": "T"}, "has_code_context": False})
    assert len(prompt) > 0

    print("[4/6] Testing sanitizer...")
    from agent.security.sanitizer import sanitize
    assert "REDACTED" in sanitize("api_key=secret123")
    assert "IP" in sanitize("192.168.1.1")
    assert sanitize("clean text") == "clean text"

    print("[5/6] Testing dry-run edge...")
    from agent.pipeline.graph import _check_dry_run
    from langgraph.graph import END
    assert _check_dry_run({"dry_run": True}) == END
    assert _check_dry_run({"dry_run": False}) == "hitl"

    print("[6/6] Testing model serialization roundtrip...")
    plan = ExplainerResult(
        summary="Add auth",
        approach="OAuth2 flow",
        steps=[PlanStep(description="Create handler", action="create", file_path="auth.py")],
        risks=["token expiry"],
        estimated_complexity="medium",
    )
    d = plan.model_dump()
    restored = ExplainerResult.model_validate(d)
    assert restored.steps[0].file_path == "auth.py"

    print("\n✅ Iteration 3 smoke test passed!")


if __name__ == "__main__":
    main()
