"""Pydantic v2 models: TaskInfo, ExplainerResult, ExecutorResult, ReviewResult, NodeDeps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from agent.providers.base import CodeProvider, DocProvider, TaskProvider


class TaskInfo(BaseModel):
    """Parsed and structured task information."""

    goal: str
    type: str = Field(description="feature | bugfix | refactor | investigation")
    domain: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    priority_comments: list[str] = Field(default_factory=list)


class PlanStep(BaseModel):
    """Single step in an execution plan."""

    description: str
    file_path: str | None = None
    action: str = Field(description="create | modify | delete | run")
    rationale: str = ""


class ExplainerResult(BaseModel):
    """Structured plan produced by the Explainer node."""

    summary: str
    approach: str
    steps: list[PlanStep] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    estimated_complexity: str = Field(default="medium", description="low | medium | high")


class FileChange(BaseModel):
    """Single file change produced by executor."""

    path: str
    action: str = Field(description="create | modify | delete")
    content: str = ""
    diff: str = ""


class ExecutorResult(BaseModel):
    """Result of code execution from the Executor node."""

    files_changed: list[FileChange] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    notes: str = ""


class ReviewFinding(BaseModel):
    """Single finding from the Reviewer."""

    severity: str = Field(description="error | warning | info")
    message: str
    file_path: str | None = None
    line: int | None = None


class ReviewResult(BaseModel):
    """Result of code review from the Reviewer node."""

    approved: bool
    findings: list[ReviewFinding] = Field(default_factory=list)
    tests_passed: bool | None = None
    static_analysis_passed: bool | None = None
    summary: str = ""


@dataclass
class NodeDeps:
    """Dependency container injected via RunContext into Pydantic AI agents."""

    task_provider: TaskProvider | None = None
    code_provider: CodeProvider | None = None
    doc_provider: DocProvider | None = None
    effects_tracker: Any = None
    task_id: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    tool_catalog: Any = None
