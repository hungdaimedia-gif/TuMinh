from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SourceType = Literal["runtime", "external"]
TaskOutcome = Literal["succeeded", "failed"]
ObservableRole = Literal["user", "assistant", "tool"]


class DistillModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @model_validator(mode="before")
    @classmethod
    def reject_private_fields(cls, value: object) -> object:
        _reject_private_fields(value)
        return value


class ImportProvenance(DistillModel):
    source_name: str = Field(min_length=1)
    license_or_authorization: str | None = None
    authorized_by: str | None = None
    imported_at: datetime
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def require_timezone(self) -> ImportProvenance:
        if self.imported_at.tzinfo is None or self.imported_at.utcoffset() is None:
            raise ValueError("imported_at must be timezone-aware")
        return self


class ObservableEvent(DistillModel):
    schema_version: Literal[1]
    task_id: str = Field(min_length=1)
    sequence_number: int = Field(ge=1)
    event_type: Literal[
        "task",
        "plan",
        "decision",
        "tool_call",
        "tool_result",
        "patch",
        "verification",
        "reflection",
        "user_message",
        "assistant_message",
        "tool_message",
    ]
    timestamp: datetime
    data: dict[str, Any]

    @model_validator(mode="after")
    def require_timezone(self) -> ObservableEvent:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("event timestamp must be timezone-aware")
        return self


class RuntimeExportHeader(DistillModel):
    record_type: Literal["header"]
    schema_version: Literal[1]
    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    source_type: SourceType
    provenance: ImportProvenance
    task_outcome: TaskOutcome
    verification_passed: bool
    eval_passed: bool | None


class TrajectoryExport(DistillModel):
    schema_version: Literal[1]
    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    source_type: SourceType
    provenance: ImportProvenance
    task_outcome: TaskOutcome
    verification_passed: bool
    eval_passed: bool | None
    events: list[ObservableEvent] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_and_events(self) -> TrajectoryExport:
        if self.source_type == "external" and (
            not self.provenance.license_or_authorization
            or not self.provenance.license_or_authorization.strip()
            or not self.provenance.authorized_by
            or not self.provenance.authorized_by.strip()
        ):
            raise ValueError(
                "External trajectories require declared authorization and authorized_by metadata"
            )
        if self.task_outcome == "succeeded" and not self.verification_passed:
            raise ValueError("A succeeded task must have passing verification")
        sequences = [event.sequence_number for event in self.events]
        if sequences != list(range(1, len(self.events) + 1)):
            raise ValueError("Trajectory event sequence numbers must be unique and contiguous")
        if any(event.task_id != self.task_id for event in self.events):
            raise ValueError("Every trajectory event must belong to the trajectory task")
        return self


class TrainingMessage(DistillModel):
    role: ObservableRole
    content: str = Field(min_length=1)
    tool_name: str | None = None

    @model_validator(mode="after")
    def require_tool_name(self) -> TrainingMessage:
        if self.role == "tool" and self.tool_name is not None and not self.tool_name.strip():
            raise ValueError("tool_name must not be empty when provided")
        return self


class TrainingRow(DistillModel):
    task_id: str = Field(min_length=1)
    messages: list[TrainingMessage] = Field(min_length=1)
    verification_passed: bool
    eval_passed: bool
    provenance_id: str = Field(min_length=1)


class DatasetManifest(DistillModel):
    dataset_version: str = Field(min_length=1)
    source_hashes: list[str]
    curation_version: str = Field(min_length=1)
    redaction_counts: dict[str, int]
    created_at: datetime
    row_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_manifest(self) -> DatasetManifest:
        if any(not re.fullmatch(r"[a-f0-9]{64}", value) for value in self.source_hashes):
            raise ValueError("source_hashes must contain lowercase SHA-256 values")
        if any(value < 0 for value in self.redaction_counts.values()):
            raise ValueError("redaction counts must be non-negative")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return self


class CandidateManifest(DistillModel):
    candidate_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    base_model_license: str = Field(min_length=1)
    dataset_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    training_config: dict[str, Any]
    runtime_metadata: dict[str, Any]
    hardware_metadata: dict[str, Any]
    benchmark_id: str = Field(min_length=1)


class DistillSchemaError(ValueError):
    """Raised for malformed trajectory export JSONL framing."""


def parse_trajectory_jsonl(content: str) -> TrajectoryExport:
    lines = content.splitlines()
    if len(lines) < 2 or any(not line.strip() for line in lines):
        raise DistillSchemaError("Trajectory export must contain a header and at least one event")
    header = RuntimeExportHeader.model_validate_json(lines[0])
    events = [ObservableEvent.model_validate_json(line) for line in lines[1:]]
    header_data = header.model_dump(exclude={"record_type"})
    return TrajectoryExport.model_validate(
        {
            **header_data,
            "events": [event.model_dump(mode="python") for event in events],
        }
    )


_PROHIBITED_FIELD_NAMES = {
    "analysis",
    "chainofthought",
    "cot",
    "developer",
    "developerprompt",
    "env",
    "environment",
    "reasoning",
    "system",
    "systemprompt",
    "thought",
    "thinking",
}
_PROHIBITED_NAME_PARTS = (
    "apikey",
    "credential",
    "password",
    "reasoning",
    "secret",
    "token",
)


def _reject_private_fields(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError("Distillation object keys must be strings")
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            if normalized == "redactioncounts" and isinstance(nested, Mapping):
                if any(
                    not isinstance(count, int) or isinstance(count, bool) or count < 0
                    for count in nested.values()
                ):
                    raise ValueError("Redaction counts must be non-negative integers")
                continue
            if (
                normalized in _PROHIBITED_FIELD_NAMES
                or normalized.startswith(
                    ("analysis", "chainofthought", "thinking", "thought", "systemprompt", "developerprompt", "environment")
                )
                or normalized.startswith("envvar")
                or any(part in normalized for part in _PROHIBITED_NAME_PARTS)
            ):
                raise ValueError(f"prohibited private or sensitive field: {key}")
            _reject_private_fields(nested)
    elif isinstance(value, list):
        for item in value:
            _reject_private_fields(item)
