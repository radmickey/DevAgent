"""Provider registry: resolves providers by name.

With MCP auto-discovery, providers are created directly from ToolCatalog
in graph.py. This module is kept for backward compatibility and testing.
"""

from __future__ import annotations

import structlog

from agent.providers.base import CodeProvider, DocProvider, TaskProvider
from agent.providers.code.stub import StubCodeProvider
from agent.providers.doc.stub import StubDocProvider
from agent.providers.task.stub import StubTaskProvider

log = structlog.get_logger()

_PROVIDERS: dict[str, dict[str, type]] = {
    "task": {"stub": StubTaskProvider},
    "code": {"stub": StubCodeProvider},
    "doc": {"stub": StubDocProvider},
}


def register_provider(kind: str, name: str, cls: type) -> None:
    """Register a provider class by kind and name."""
    if kind not in _PROVIDERS:
        _PROVIDERS[kind] = {}
    _PROVIDERS[kind][name] = cls


def get_stub_providers() -> tuple[TaskProvider, CodeProvider, DocProvider]:
    """Return stub providers for all three types."""
    return StubTaskProvider(), StubCodeProvider(), StubDocProvider()
