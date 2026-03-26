"""DevAgent configuration: env loading, LLM setup, timeouts."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_devagent_home() -> Path:
    p = Path(os.getenv("DEVAGENT_HOME", str(Path.home() / ".devagent")))
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return p


DEVAGENT_HOME = _get_devagent_home()

TIMEOUTS: dict[str, int] = {
    "mcp_read": 30,
    "mcp_search": 20,
    "mcp_write": 45,
    "llm_fast": 30,
    "llm_strong": 120,
}


@dataclass(frozen=True)
class Config:
    task_provider: str = field(
        default_factory=lambda: os.getenv("TASK_PROVIDER", "yandex_tracker")
    )
    code_provider: str = field(
        default_factory=lambda: os.getenv("CODE_PROVIDER", "github")
    )
    doc_provider: str = field(
        default_factory=lambda: os.getenv("DOC_PROVIDER", "notion")
    )

    llm_strong_model: str = field(
        default_factory=lambda: os.getenv("LLM_STRONG_MODEL", "claude-sonnet-4-20250514")
    )
    llm_fast_model: str = field(
        default_factory=lambda: os.getenv("LLM_FAST_MODEL", "claude-haiku-4-20250414")
    )

    use_external_llm: bool = field(
        default_factory=lambda: os.getenv("USE_EXTERNAL_LLM", "true").lower() == "true"
    )

    env: str = field(
        default_factory=lambda: os.getenv("DEVAGENT_ENV", "dev")
    )

    home: Path = field(default_factory=_get_devagent_home)


def get_config() -> Config:
    return Config()
