"""Iteration 1 smoke test."""

import os

from agent.config import get_config
from agent.errors import TransientError, PermanentError, DegradedError
from agent.utils.lockfile import AgentLock
from agent.utils.timeout import with_timeout
from agent.pipeline.graph import build_pipeline
from agent.pipeline.state import PipelineState
from agent.providers.registry import build_providers

config = get_config()
print("Config OK:", config)

graph, cfg = build_pipeline()
print("Graph OK")

lock_path = os.path.expanduser("~/.devagent/agent.lock")
if os.path.exists(lock_path):
    os.remove(lock_path)
with AgentLock():
    print("Lockfile OK")
print("Lockfile released OK")

print()
print("=== ITERATION 1 SMOKE TEST: PASS ===")
