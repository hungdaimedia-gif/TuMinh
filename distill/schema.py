from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from math import isclose, isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


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
    source_model_id: str | None = Field(default=None, min_length=1)
    source_model_revision: str | None = Field(default=None, min_length=1)
    imported_at: datetime
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def require_timezone(self) -> ImportProvenance:
        if self.imported_at.tzinfo is None or self.imported_at.utcoffset() is None:
            raise ValueError("imported_at must be timezone-aware")
        return self

    @model_validator(mode="after")
    def require_source_model_for_revision(self) -> ImportProvenance:
        if self.source_model_revision is not None and self.source_model_id is None:
            raise ValueError("source_model_revision requires source_model_id")
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


class ExternalExportHeader(DistillModel):
    record_type: Literal["header"]
    schema_version: Literal[1]
    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
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
    source_provenance: list[ImportProvenance] = Field(default_factory=list)
    curation_version: str = Field(min_length=1)
    redaction_counts: dict[str, int]
    created_at: datetime
    row_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_manifest(self) -> DatasetManifest:
        if any(not re.fullmatch(r"[a-f0-9]{64}", value) for value in self.source_hashes):
            raise ValueError("source_hashes must contain lowercase SHA-256 values")
        if any(
            provenance.source_sha256 not in self.source_hashes
            for provenance in self.source_provenance
        ):
            raise ValueError("source_provenance must correspond to a source hash")
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


class EvalCase(DistillModel):
    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    safety_critical: StrictBool


class ModelScore(DistillModel):
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    case_results: dict[str, StrictBool] = Field(min_length=1)
    passed_count: int = Field(ge=0)
    total_count: int = Field(ge=1)
    pass_rate: float = Field(ge=0, le=1)
    resource_measurements: dict[str, float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_score(self) -> ModelScore:
        if any(not key for key in self.case_results):
            raise ValueError("case result IDs must not be empty")
        if self.total_count != len(self.case_results):
            raise ValueError("total_count must equal the number of case results")
        if self.passed_count != sum(self.case_results.values()):
            raise ValueError("passed_count must equal the number of passing cases")
        if not isclose(
            self.pass_rate,
            self.passed_count / self.total_count,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("pass_rate must match the reported case results")
        if any(
            not key or not isfinite(value) or value < 0
            for key, value in self.resource_measurements.items()
        ):
            raise ValueError("resource measurements must be named, finite, non-negative values")
        return self

    @property
    def failed_case_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                case_id
                for case_id, passed in self.case_results.items()
                if not passed
            )
        )



class BenchmarkMetadata(DistillModel):
    benchmark_id: str | None = None
    baseline_model_id: str | None = None
    baseline_model_revision: str | None = None
    candidate_model_id: str | None = None
    candidate_model_revision: str | None = None
    dataset_hash: str | None = None
    eval_set_version: str | None = None
    safety_critical_case_ids: list[str] = Field(default_factory=list)
    tool_limits: dict[str, Any] = Field(default_factory=dict)
    decoding_config: dict[str, Any] = Field(default_factory=dict)
    software_metadata: dict[str, Any] = Field(default_factory=dict)
    hardware_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dataset_hash(self) -> BenchmarkMetadata:
        if self.dataset_hash is not None and not re.fullmatch(
            r"[a-f0-9]{64}", self.dataset_hash
        ):
            raise ValueError("dataset_hash must be a lowercase SHA-256 value")
        return self

    @property
    def complete(self) -> bool:
        identifiers = (
            self.benchmark_id,
            self.baseline_model_id,
            self.baseline_model_revision,
            self.candidate_model_id,
            self.candidate_model_revision,
            self.dataset_hash,
            self.eval_set_version,
        )
        configurations = (
            self.tool_limits,
            self.decoding_config,
            self.software_metadata,
            self.hardware_metadata,
        )
        return (
            all(isinstance(value, str) and bool(value.strip()) for value in identifiers)
            and isinstance(self.dataset_hash, str)
            and bool(re.fullmatch(r"[a-f0-9]{64}", self.dataset_hash))
            and all(isinstance(config, dict) and config for config in configurations)
            and bool(self.safety_critical_case_ids)
            and all(
                isinstance(case_id, str) and case_id.strip()
                for case_id in self.safety_critical_case_ids
            )
            and len(set(self.safety_critical_case_ids))
            == len(self.safety_critical_case_ids)
        )


class BenchmarkReport(DistillModel):
    metadata: BenchmarkMetadata
    baseline_score: ModelScore
    candidate_score: ModelScore
    eval_set_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    eval_case_ids: list[str] = Field(min_length=1)
    safety_case_ids: list[str]
    safety_results: dict[str, StrictBool]

    @property
    def eligible_for_promotion(self) -> bool:
        expected_case_ids = set(self.eval_case_ids)
        if (
            not self.metadata.complete
            or len(expected_case_ids) != len(self.eval_case_ids)
            or not self.safety_case_ids
            or not set(self.safety_case_ids) <= expected_case_ids
            or set(self.safety_case_ids) != set(self.safety_results)
            or set(self.safety_case_ids) != set(self.metadata.safety_critical_case_ids)
            or len(set(self.safety_case_ids)) != len(self.safety_case_ids)
            or any(
                not isinstance(result, bool) or not result
                for result in self.safety_results.values()
            )
            or set(self.baseline_score.case_results) != expected_case_ids
            or set(self.candidate_score.case_results) != expected_case_ids
            or self.baseline_score.total_count != len(self.eval_case_ids)
            or self.candidate_score.total_count != len(self.eval_case_ids)
            or not self.baseline_score.resource_measurements
            or not self.candidate_score.resource_measurements
        ):
            return False
        return (
            self.baseline_score.model_id == self.metadata.baseline_model_id
            and self.baseline_score.model_revision == self.metadata.baseline_model_revision
            and self.candidate_score.model_id == self.metadata.candidate_model_id
            and self.candidate_score.model_revision == self.metadata.candidate_model_revision
            and all(
                self.candidate_score.case_results[case_id] == result
                for case_id, result in self.safety_results.items()
            )
            and self.candidate_score.pass_rate >= self.baseline_score.pass_rate
        )


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
