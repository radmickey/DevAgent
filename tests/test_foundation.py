"""Tests for Task 1.1: config, lockfile, timeout, errors."""

from __future__ import annotations

import asyncio
import os

import pytest

from agent.config import Config, get_config, DEVAGENT_HOME, TIMEOUTS
from agent.errors import DevAgentError, TransientError, PermanentError, DegradedError
from agent.utils.lockfile import AgentLock
from agent.utils.timeout import with_timeout


class TestConfig:
    def test_get_config_returns_config(self):
        cfg = get_config()
        assert isinstance(cfg, Config)

    def test_config_defaults(self):
        cfg = get_config()
        assert cfg.home == DEVAGENT_HOME
        assert cfg.env in ("dev", "prod")

    def test_config_has_llm_models(self):
        cfg = get_config()
        assert cfg.llm_strong_model
        assert cfg.llm_fast_model

    def test_timeouts_defined(self):
        assert "mcp_read" in TIMEOUTS
        assert "mcp_search" in TIMEOUTS
        assert "mcp_write" in TIMEOUTS
        assert "llm_fast" in TIMEOUTS
        assert "llm_strong" in TIMEOUTS


class TestErrors:
    def test_transient_error(self):
        err = TransientError("timeout")
        assert str(err) == "timeout"

    def test_permanent_error(self):
        err = PermanentError("not found")
        assert str(err) == "not found"

    def test_degraded_error(self):
        err = DegradedError("service down")
        assert str(err) == "service down"

    def test_error_hierarchy(self):
        assert issubclass(TransientError, DevAgentError)
        assert issubclass(PermanentError, DevAgentError)
        assert issubclass(DegradedError, DevAgentError)


class TestLockfile:
    @pytest.fixture(autouse=True)
    def cleanup_lock(self):
        lock_path = AgentLock.LOCK_PATH
        if lock_path.exists():
            lock_path.unlink()
        yield
        if lock_path.exists():
            lock_path.unlink()

    def test_lock_creates_and_removes_file(self):
        with AgentLock():
            assert AgentLock.LOCK_PATH.exists()
            content = AgentLock.LOCK_PATH.read_text().strip()
            assert content == str(os.getpid())
        assert not AgentLock.LOCK_PATH.exists()

    def test_lock_prevents_second_instance(self):
        with AgentLock():
            with pytest.raises(RuntimeError, match="already running"):
                with AgentLock():
                    pass

    def test_lock_removes_stale_lockfile(self):
        AgentLock.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        AgentLock.LOCK_PATH.write_text("999999999")
        with AgentLock():
            assert AgentLock.LOCK_PATH.exists()

    def test_lock_removes_corrupt_lockfile(self):
        AgentLock.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        AgentLock.LOCK_PATH.write_text("not-a-pid")
        with AgentLock():
            assert AgentLock.LOCK_PATH.exists()


class TestTimeout:
    @pytest.mark.asyncio
    async def test_with_timeout_success(self):
        async def fast():
            return 42
        result = await with_timeout(fast(), timeout=5, name="test")
        assert result == 42

    @pytest.mark.asyncio
    async def test_with_timeout_raises_transient(self):
        async def slow():
            await asyncio.sleep(10)
        with pytest.raises(TransientError, match="timed out"):
            await with_timeout(slow(), timeout=0.1, name="slow_call")
