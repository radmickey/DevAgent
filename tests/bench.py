"""Benchmark: run 20 synthetic tasks and measure time/tokens/cost/quality.

Usage:
    python tests/bench.py [--tasks N] [--output results.json]

Runs the pipeline in dry-run mode on synthetic tasks and records metrics.
Requires ANTHROPIC_API_KEY or local Ollama to be configured.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

SYNTHETIC_TASKS: list[dict[str, str]] = [
    {"id": "BENCH-1", "title": "Add OAuth2 login flow", "type": "feature"},
    {"id": "BENCH-2", "title": "Fix N+1 query in user list endpoint", "type": "bugfix"},
    {"id": "BENCH-3", "title": "Refactor payment module to use Strategy pattern", "type": "refactor"},
    {"id": "BENCH-4", "title": "Add rate limiting to public API", "type": "feature"},
    {"id": "BENCH-5", "title": "Fix timezone handling in scheduled tasks", "type": "bugfix"},
    {"id": "BENCH-6", "title": "Migrate from REST to GraphQL for dashboard API", "type": "refactor"},
    {"id": "BENCH-7", "title": "Add WebSocket support for real-time notifications", "type": "feature"},
    {"id": "BENCH-8", "title": "Fix memory leak in background worker", "type": "bugfix"},
    {"id": "BENCH-9", "title": "Add RBAC (role-based access control)", "type": "feature"},
    {"id": "BENCH-10", "title": "Optimize database indexing for search queries", "type": "investigation"},
    {"id": "BENCH-11", "title": "Add CSV export for analytics data", "type": "feature"},
    {"id": "BENCH-12", "title": "Fix race condition in order processing", "type": "bugfix"},
    {"id": "BENCH-13", "title": "Refactor logging to use structured format", "type": "refactor"},
    {"id": "BENCH-14", "title": "Add health check endpoint with dependencies", "type": "feature"},
    {"id": "BENCH-15", "title": "Fix SSL certificate validation in HTTP client", "type": "bugfix"},
    {"id": "BENCH-16", "title": "Add pagination to all list endpoints", "type": "feature"},
    {"id": "BENCH-17", "title": "Investigate slow CI pipeline", "type": "investigation"},
    {"id": "BENCH-18", "title": "Add retry logic for external API calls", "type": "feature"},
    {"id": "BENCH-19", "title": "Fix broken email templates after migration", "type": "bugfix"},
    {"id": "BENCH-20", "title": "Refactor configuration to use 12-factor app pattern", "type": "refactor"},
]


@dataclass
class TaskResult:
    task_id: str
    title: str
    task_type: str
    elapsed_ms: float = 0.0
    plan_steps: int = 0
    plan_complexity: str = ""
    has_plan: bool = False
    error: str = ""


@dataclass
class BenchmarkReport:
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_tasks: int = 0
    successful: int = 0
    failed: int = 0
    avg_elapsed_ms: float = 0.0
    avg_plan_steps: float = 0.0
    results: list[dict[str, Any]] = field(default_factory=list)


async def run_single_task(task: dict[str, str]) -> TaskResult:
    """Run a single task through the pipeline in dry-run mode."""
    from agent.pipeline.nodes.explainer import explainer_node
    from agent.pipeline.nodes.input_router import input_router_node

    result = TaskResult(
        task_id=task["id"],
        title=task["title"],
        task_type=task["type"],
    )

    state = {
        "task_id": task["id"],
        "task_raw": {"title": task["title"], "description": f"Type: {task['type']}"},
        "enriched_context": {},
        "ranked_context": [],
        "has_code_context": False,
        "has_doc_context": False,
        "has_similar_tasks": False,
        "enrichment_warnings": [],
        "dry_run": True,
    }

    start = time.perf_counter()
    try:
        state = await input_router_node(state)
        state = await explainer_node(state)

        plan = state.get("plan", {})
        result.has_plan = bool(plan)
        result.plan_steps = len(plan.get("steps", []))
        result.plan_complexity = plan.get("estimated_complexity", "unknown")
    except Exception as exc:
        result.error = str(exc)

    result.elapsed_ms = (time.perf_counter() - start) * 1000
    return result


async def run_benchmark(tasks: list[dict[str, str]] | None = None) -> BenchmarkReport:
    """Run benchmark on all tasks sequentially."""
    tasks = tasks or SYNTHETIC_TASKS
    report = BenchmarkReport(total_tasks=len(tasks))
    results: list[TaskResult] = []

    for i, task in enumerate(tasks, 1):
        log.info("bench_task", index=i, total=len(tasks), task_id=task["id"])
        result = await run_single_task(task)
        results.append(result)
        if result.error:
            report.failed += 1
        else:
            report.successful += 1

    report.results = [asdict(r) for r in results]

    successful = [r for r in results if not r.error]
    if successful:
        report.avg_elapsed_ms = sum(r.elapsed_ms for r in successful) / len(successful)
        report.avg_plan_steps = sum(r.plan_steps for r in successful) / len(successful)

    return report


def main() -> None:
    """Run benchmark and save results."""
    import argparse

    parser = argparse.ArgumentParser(description="DevAgent Benchmark")
    parser.add_argument("--tasks", type=int, default=20, help="Number of tasks to run")
    parser.add_argument("--output", type=str, default="bench_results.json", help="Output file")
    args = parser.parse_args()

    from agent.logging import setup_logging
    setup_logging()

    tasks = SYNTHETIC_TASKS[:args.tasks]
    report = asyncio.run(run_benchmark(tasks))

    output_path = Path(args.output)
    output_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"Benchmark complete: {report.successful}/{report.total_tasks} successful")
    print(f"Average time: {report.avg_elapsed_ms:.0f}ms")
    print(f"Average plan steps: {report.avg_plan_steps:.1f}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
