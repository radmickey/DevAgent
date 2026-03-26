"""MultiProvider: wraps multiple providers with fallback/round-robin/priority strategies.

Strategies:
- fallback: try each provider in order, use first that succeeds
- round_robin: distribute calls across providers evenly
- priority: always use first provider, fall back on failure
"""

from __future__ import annotations

import itertools
from typing import Any

import structlog

from agent.providers.base import CodeProvider, DocProvider, TaskProvider

log = structlog.get_logger()


class MultiTaskProvider(TaskProvider):
    """Multi-provider wrapper for TaskProvider with configurable strategy."""

    def __init__(
        self,
        providers: list[TaskProvider],
        strategy: str = "fallback",
    ) -> None:
        if not providers:
            raise ValueError("At least one provider is required")
        self._providers = providers
        self._strategy = strategy
        self._rr_iter = itertools.cycle(range(len(providers)))

    async def get_task(self, task_id: str) -> dict[str, Any]:
        result: dict[str, Any] = await self._call("get_task", task_id)
        return result

    async def get_comments(self, task_id: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = await self._call("get_comments", task_id)
        return result

    async def update_status(self, task_id: str, status: str) -> None:
        await self._call("update_status", task_id, status)

    async def _call(self, method: str, *args: Any) -> Any:
        if self._strategy == "round_robin":
            return await self._call_round_robin(method, *args)
        return await self._call_fallback(method, *args)

    async def _call_fallback(self, method: str, *args: Any) -> Any:
        last_exc: Exception | None = None
        for i, provider in enumerate(self._providers):
            try:
                result = await getattr(provider, method)(*args)
                if i > 0:
                    log.info("multi_fallback_used", provider=i, method=method)
                return result
            except Exception as exc:
                last_exc = exc
                log.warning("multi_provider_failed", provider=i, method=method, error=str(exc))
        raise last_exc or RuntimeError("All providers failed")

    async def _call_round_robin(self, method: str, *args: Any) -> Any:
        idx = next(self._rr_iter)
        try:
            return await getattr(self._providers[idx], method)(*args)
        except Exception:
            return await self._call_fallback(method, *args)


class MultiCodeProvider(CodeProvider):
    """Multi-provider wrapper for CodeProvider."""

    def __init__(self, providers: list[CodeProvider], strategy: str = "fallback") -> None:
        if not providers:
            raise ValueError("At least one provider is required")
        self._providers = providers
        self._strategy = strategy
        self._rr_iter = itertools.cycle(range(len(providers)))

    async def search_code(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = await self._call("search_code", query, limit)
        return result

    async def get_file(self, path: str, ref: str = "main") -> str:
        result: str = await self._call("get_file", path, ref)
        return result

    async def list_files(self, path: str = "", ref: str = "main") -> list[str]:
        result: list[str] = await self._call("list_files", path, ref)
        return result

    async def get_diff(self, base: str, head: str) -> str:
        result: str = await self._call("get_diff", base, head)
        return result

    async def _call(self, method: str, *args: Any) -> Any:
        if self._strategy == "round_robin":
            idx = next(self._rr_iter)
            try:
                return await getattr(self._providers[idx], method)(*args)
            except Exception:
                pass

        last_exc: Exception | None = None
        for i, provider in enumerate(self._providers):
            try:
                return await getattr(provider, method)(*args)
            except Exception as exc:
                last_exc = exc
                log.warning("multi_code_failed", provider=i, method=method, error=str(exc))
        raise last_exc or RuntimeError("All code providers failed")


class MultiDocProvider(DocProvider):
    """Multi-provider wrapper for DocProvider."""

    def __init__(self, providers: list[DocProvider], strategy: str = "fallback") -> None:
        if not providers:
            raise ValueError("At least one provider is required")
        self._providers = providers
        self._strategy = strategy
        self._rr_iter = itertools.cycle(range(len(providers)))

    async def search_docs(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = await self._call("search_docs", query, limit)
        return result

    async def get_page(self, page_id: str) -> dict[str, Any]:
        result: dict[str, Any] = await self._call("get_page", page_id)
        return result

    async def _call(self, method: str, *args: Any) -> Any:
        if self._strategy == "round_robin":
            idx = next(self._rr_iter)
            try:
                return await getattr(self._providers[idx], method)(*args)
            except Exception:
                pass

        last_exc: Exception | None = None
        for i, provider in enumerate(self._providers):
            try:
                return await getattr(provider, method)(*args)
            except Exception as exc:
                last_exc = exc
                log.warning("multi_doc_failed", provider=i, method=method, error=str(exc))
        raise last_exc or RuntimeError("All doc providers failed")
