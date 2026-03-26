"""Timeout wrapper for all external calls."""

from __future__ import annotations

import asyncio

from agent.errors import TransientError


async def with_timeout(
    coro,  # noqa: ANN001
    timeout: int = 30,
    name: str = "call",
):
    """Wrap an awaitable with asyncio.wait_for; raise TransientError on timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TransientError(f"{name} timed out after {timeout}s")
