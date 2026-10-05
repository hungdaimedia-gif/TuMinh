from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Task(DomainModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    acceptance_criteria: list[str] = Field(default_factory=list)
    source: Literal["local", "github"]
    state: Literal["pending", "running", "succeeded", "failed"] = "pending"


class Subtask(DomainModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    files: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)


class TaskPlan(DomainModel):
    task_id: str = Field(min_length=1)
    subtasks: list[Subtask] = Field(min_length=1)
    verification_commands: list[list[str]] = Field(default_factory=list)


class ToolCall(DomainModel):
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class AgentDecision(DomainModel):
    tool_call: ToolCall | None = None
    final_response: str | None = None

    @model_validator(mode="after")
    def require_exactly_one_outcome(self) -> AgentDecision:
        if (self.tool_call is None) == (self.final_response is None):
            raise ValueError("Exactly one of tool_call or final_response is required")
        if self.final_response is not None and not self.final_response.strip():
            raise ValueError("final_response must not be empty")
        return self


class TaskResult(DomainModel):
    task_id: str = Field(min_length=1)
    status: Literal["succeeded", "failed"]
    summary: str
    changed_files: list[str] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="after")
    def validate_status_details(self) -> TaskResult:
        if self.status == "failed" and not self.error:
            raise ValueError("Failed tasks must include an error reason")
        if self.status == "succeeded" and self.error is not None:
            raise ValueError("Successful tasks cannot include an error reason")
        return self


class VerificationResult(DomainModel):
    passed: bool
    commands: list[list[str]] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_failure_details(self) -> VerificationResult:
        if self.passed and self.failures:
            raise ValueError("Passing verification cannot include failures")
        if not self.passed and not self.failures:
            raise ValueError("Failed verification must include a failure reason")
        return self

    @property
    def failure_summary(self) -> str:
        if self.passed:
            return "All verification commands passed."
        return f"{len(self.failures)} verification failure(s): {'; '.join(self.failures)}"



class Reflection(DomainModel):
    task_id: str = Field(min_length=1)
    outcome: Literal["succeeded", "failed"]
    lessons: list[str] = Field(default_factory=list)
    failure_category: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
