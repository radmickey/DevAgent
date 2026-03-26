"""Smoke test for Iteration 4: observability, meta-agent, doc_writer, bench."""

from __future__ import annotations


def main():
    print("[1/6] Observability module...")
    from agent.observability import get_tracing_status
    status = get_tracing_status()
    assert isinstance(status, dict)

    print("[2/6] Meta-Agent module...")
    from agent.pipeline.meta_agent import PromptUpdate
    update = PromptUpdate(node="test", new_prompt="p", reason="r", confidence=0.5)
    assert update.node == "test"

    print("[3/6] DocWriter node...")
    from agent.pipeline.nodes.doc_writer import (
        _build_summary,
        _detect_languages,
    )
    summary = _build_summary({"title": "T"}, {}, [], {})
    assert "T" in summary
    assert _detect_languages([{"path": "a.py"}]) == ["python"]

    print("[4/6] pyproject.toml structure...")
    import tomllib
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    assert data["project"]["scripts"]["devagent"] == "agent.interface.cli:main"
    assert "dev" in data["project"]["optional-dependencies"]

    print("[5/6] Benchmark structure...")
    from tests.bench import SYNTHETIC_TASKS, BenchmarkReport
    assert len(SYNTHETIC_TASKS) == 20
    report = BenchmarkReport(total_tasks=1, successful=1)
    assert report.successful == 1

    print("[6/6] Vector memory lang filter...")
    from agent.memory.vector import VectorMemory
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as td:
        vm = VectorMemory(persist_dir=os.path.join(td, "chroma"))
        vm.store("T-1", [{"content": "test", "source": "test", "lang": "python"}])
        results = vm.query("T-1", "test", lang="python")
        assert len(results) >= 1

    print("\n✅ Iteration 4 smoke test passed!")


if __name__ == "__main__":
    main()
