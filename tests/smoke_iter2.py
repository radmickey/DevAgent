"""Iteration 2 smoke test."""

import asyncio
import tempfile

from agent.memory.vector import VectorMemory
from agent.memory.budget import fit_context, count_tokens
from agent.memory.cost import CostTracker

with tempfile.TemporaryDirectory() as tmpdir:
    vm = VectorMemory(persist_dir=tmpdir)
    vm.store("test_task", [{"source": "code", "content": "def hello(): pass"}])
    results = vm.query("test_task", "hello function", max_tokens=500)
    print("Vector memory OK:", len(results), "results")

tokens = count_tokens("Hello, world!")
print("tiktoken OK:", tokens, "tokens")

with tempfile.TemporaryDirectory() as tmpdir:
    from pathlib import Path
    ct = CostTracker(db_path=Path(tmpdir) / "cost.db")
    ct.log(task_id="TEST-1", node="explainer", model="claude-sonnet",
           tokens_in=100, tokens_out=50, latency_ms=500, cost_usd=0.001)
    print("Cost tracker OK")
    summary = ct.get_summary_by_model()
    print("Cost summary by model OK:", summary)
    ct.close()

print()
print("=== ITERATION 2 SMOKE TEST: PASS ===")
