"""Web search provider: Tavily API with optional include_domains filtering."""

from __future__ import annotations

import os
from typing import Any

import structlog

from agent.errors import DegradedError, TransientError

log = structlog.get_logger()


class WebSearchProvider:
    """Web search via Tavily API. Falls back gracefully if unavailable."""

    def __init__(
        self,
        api_key: str | None = None,
        include_domains: list[str] | None = None,
        max_results: int = 5,
    ) -> None:
        self._api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self._include_domains = include_domains or []
        self._max_results = max_results

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    async def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """Search the web using Tavily.

        Returns list of dicts with keys: title, url, content, score.
        """
        if not self._api_key:
            raise DegradedError("Tavily API key not configured (TAVILY_API_KEY)")

        n = max_results or self._max_results

        try:
            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(api_key=self._api_key)
            kwargs: dict[str, Any] = {
                "query": query,
                "max_results": n,
                "search_depth": "basic",
            }
            if self._include_domains:
                kwargs["include_domains"] = self._include_domains

            response = await client.search(**kwargs)
            results = response.get("results", [])

            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                    "source": "web_search",
                }
                for r in results
            ]
        except ImportError:
            raise DegradedError("tavily-python not installed: pip install tavily-python")
        except Exception as exc:
            raise TransientError(f"Tavily search failed: {exc}")

    async def search_safe(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """Search with graceful degradation — returns empty list on failure."""
        try:
            return await self.search(query, max_results)
        except (DegradedError, TransientError) as exc:
            log.warning("web_search_degraded", query=query[:50], error=str(exc))
            return []
