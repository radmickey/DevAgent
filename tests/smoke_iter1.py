"""Iteration 1 smoke test."""

import os

from agent.config import get_config
from agent.utils.lockfile import AgentLock
from agent.pipeline.graph import build_pipeline

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
