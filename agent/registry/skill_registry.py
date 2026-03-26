"""Skill registry: Python dict-based skill discovery."""

from __future__ import annotations

from typing import Callable

import structlog

log = structlog.get_logger()

_SKILLS: dict[str, Callable] = {}


def register_skill(name: str, func: Callable) -> None:
    """Register a skill function by name."""
    _SKILLS[name] = func
    log.info("skill_registered", name=name)


def get_skill(name: str) -> Callable | None:
    """Get a skill by name."""
    return _SKILLS.get(name)


def list_skills() -> list[str]:
    """List all registered skill names."""
    return list(_SKILLS.keys())
