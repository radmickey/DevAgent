"""Provider registry: build_providers() from .env configuration."""

from __future__ import annotations

import structlog

from agent.config import Config, get_config
from agent.errors import PermanentError
from agent.providers.base import CodeProvider, DocProvider, TaskProvider
from agent.providers.code.stub import StubCodeProvider
from agent.providers.doc.stub import StubDocProvider
from agent.providers.task.stub import StubTaskProvider

log = structlog.get_logger()

_TASK_PROVIDERS: dict[str, type[TaskProvider]] = {
    "stub": StubTaskProvider,
}

_CODE_PROVIDERS: dict[str, type[CodeProvider]] = {
    "stub": StubCodeProvider,
}

_DOC_PROVIDERS: dict[str, type[DocProvider]] = {
    "stub": StubDocProvider,
}


def register_task_provider(name: str, cls: type[TaskProvider]) -> None:
    _TASK_PROVIDERS[name] = cls


def register_code_provider(name: str, cls: type[CodeProvider]) -> None:
    _CODE_PROVIDERS[name] = cls


def register_doc_provider(name: str, cls: type[DocProvider]) -> None:
    _DOC_PROVIDERS[name] = cls


def build_providers(
    config: Config | None = None,
) -> tuple[TaskProvider, CodeProvider, DocProvider]:
    """Instantiate one provider per type based on config."""
    if config is None:
        config = get_config()

    task_name = config.task_provider
    code_name = config.code_provider
    doc_name = config.doc_provider

    if task_name not in _TASK_PROVIDERS:
        raise PermanentError(
            f"Unknown task provider: {task_name}. "
            f"Available: {list(_TASK_PROVIDERS.keys())}"
        )
    if code_name not in _CODE_PROVIDERS:
        raise PermanentError(
            f"Unknown code provider: {code_name}. "
            f"Available: {list(_CODE_PROVIDERS.keys())}"
        )
    if doc_name not in _DOC_PROVIDERS:
        raise PermanentError(
            f"Unknown doc provider: {doc_name}. "
            f"Available: {list(_DOC_PROVIDERS.keys())}"
        )

    task_prov = _TASK_PROVIDERS[task_name]()
    code_prov = _CODE_PROVIDERS[code_name]()
    doc_prov = _DOC_PROVIDERS[doc_name]()

    log.info(
        "providers_built",
        task=task_name,
        code=code_name,
        doc=doc_name,
    )
    return task_prov, code_prov, doc_prov
