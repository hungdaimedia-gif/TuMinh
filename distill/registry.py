from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback below
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX only
    msvcrt = None

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from distill.schema import BenchmarkReport, CandidateManifest


_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_MODEL_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class RegistryError(ValueError):
    """Raised when local registry state is invalid or cannot be persisted."""


class PromotionError(RegistryError):
    """Raised when promotion or rollback does not satisfy its safety gates."""


class HumanApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    approved_by: str
    confirmed_at: datetime

    @field_validator("approved_by")
    @classmethod
    def require_approver(cls, value: str) -> str:
        if not value:
            raise ValueError("approved_by must not be empty")
        return value

    @field_validator("confirmed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("confirmed_at must be timezone-aware")
        return value


class RegistryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    active_model: str | None = None
    previous_model: str | None = None
    initial_model: str | None = None
    initialized_baseline_model_id: str | None = None
    initialized_baseline_revision: str | None = None
    installed_models: list[str] = Field(default_factory=list)
    candidates: dict[str, CandidateManifest] = Field(default_factory=dict)
    promoted_at: datetime | None = None

    @model_validator(mode="after")
    def validate_pointers(self) -> RegistryState:
        if len(self.installed_models) != len(set(self.installed_models)):
            raise ValueError("installed model tags must be unique")
        if any(not _is_valid_model_tag(tag) for tag in self.installed_models):
            raise ValueError("installed model tag is invalid")
        if self.initial_model is not None and (
            not _is_valid_model_tag(self.initial_model)
            or self.initial_model not in self.installed_models
        ):
            raise ValueError("initial model must be a valid installed tag")
        if self.initialized_baseline_model_id is not None and not self.initialized_baseline_model_id.strip():
            raise ValueError("initialized baseline model ID must not be empty")
        if self.initialized_baseline_revision is not None and not self.initialized_baseline_revision.strip():
            raise ValueError("initialized baseline revision must not be empty")
        if (self.initialized_baseline_model_id is None) != (self.initialized_baseline_revision is None):
            raise ValueError("initialized baseline model ID and revision must be provided together")
        for tag in (self.active_model, self.previous_model):
            if tag is not None and tag not in self.installed_models:
                raise ValueError("active and previous models must be installed")
            if (
                tag is not None
                and tag != self.initial_model
                and tag not in self.candidates
            ):
                raise ValueError("active and previous models must reference known versions")
        if self.active_model is not None and self.active_model == self.previous_model:
            raise ValueError("active and previous models must be distinct")
        for candidate_id, manifest in self.candidates.items():
            if not _is_valid_candidate_id(candidate_id):
                raise ValueError("candidate registry key is invalid")
            if manifest.candidate_id != candidate_id:
                raise ValueError("candidate registry key must match its manifest")
            if candidate_id not in self.installed_models:
                raise ValueError("registered candidates must be installed")
        if self.promoted_at is not None and (
            self.promoted_at.tzinfo is None or self.promoted_at.utcoffset() is None
        ):
            raise ValueError("promoted_at must be timezone-aware")
        return self


class DistillModelRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)

    def register_candidate(self, manifest: CandidateManifest) -> None:
        try:
            candidate = CandidateManifest.model_validate(
                manifest.model_dump(mode="python")
            )
        except Exception as error:
            raise RegistryError("candidate manifest is invalid") from error
        if not _is_valid_candidate_id(candidate.candidate_id):
            raise RegistryError("candidate ID is invalid")

        with self._locked_registry():
            state = self.read()
            existing = state.candidates.get(candidate.candidate_id)
            if existing is not None:
                if existing.model_dump(mode="json") == candidate.model_dump(mode="json"):
                    return
                raise RegistryError("candidate ID is already registered with different metadata")

            candidates = dict(state.candidates)
            candidates[candidate.candidate_id] = candidate
            installed_models = list(state.installed_models)
            if candidate.candidate_id not in installed_models:
                installed_models.append(candidate.candidate_id)
            self._write(
                state.model_copy(
                    update={"candidates": candidates, "installed_models": installed_models}
                )
            )

    def initialize_active(
        self,
        model_tag: str,
        baseline_model_id: str | None = None,
        baseline_model_revision: str | None = None,
    ) -> None:
        if not _is_valid_model_tag(model_tag):
            raise RegistryError("active model tag is invalid")
        if baseline_model_id is not None and not baseline_model_id.strip():
            raise RegistryError("baseline model ID is invalid")
        if baseline_model_revision is not None and not baseline_model_revision.strip():
            raise RegistryError("baseline model revision is invalid")
        if (baseline_model_id is None) != (baseline_model_revision is None):
            raise RegistryError("baseline model ID and revision must be provided together")
        with self._locked_registry():
            state = self.read()
            if state.active_model is not None or state.initial_model is not None:
                raise RegistryError("active model is already initialized")
            installed_models = list(state.installed_models)
            if model_tag not in installed_models:
                installed_models.append(model_tag)
            self._write(
                state.model_copy(
                    update={
                        "active_model": model_tag,
                        "initial_model": model_tag,
                        "initialized_baseline_model_id": baseline_model_id,
                        "initialized_baseline_revision": baseline_model_revision,
                        "installed_models": installed_models,
                    }
                )
            )

    def read(self) -> RegistryState:
        if not self.path.exists():
            return RegistryState(schema_version=1)
        try:
            raw = self.path.read_text(encoding="utf-8")
            state = RegistryState.model_validate_json(raw)
        except Exception as error:
            raise RegistryError("registry file is invalid") from error
        return state.model_copy(deep=True)

    def promote(
        self,
        candidate_id: str,
        report: BenchmarkReport,
        approval: HumanApproval,
    ) -> RegistryState:
        if not _is_valid_candidate_id(candidate_id):
            raise PromotionError("candidate ID is invalid")
        if approval is None:
            raise PromotionError("human approval is required")
        try:
            validated_approval = HumanApproval.model_validate(
                approval.model_dump(mode="python")
            )
            validated_report = BenchmarkReport.model_validate(
                report.model_dump(mode="python")
            )
        except Exception as error:
            raise PromotionError("benchmark report or human approval is invalid") from error

        with self._locked_registry():
            state = self.read()
            manifest = state.candidates.get(candidate_id)
            if manifest is None:
                raise PromotionError("candidate is not registered")
            if state.active_model is None:
                raise PromotionError("registry has no active model")
            if not validated_report.eligible_for_promotion:
                raise PromotionError("benchmark report is not eligible for promotion")
            metadata = validated_report.metadata
            if (
                not metadata.complete
                or metadata.benchmark_id != manifest.benchmark_id
                or metadata.candidate_model_id != manifest.model_id
                or metadata.candidate_model_revision != manifest.model_revision
                or metadata.dataset_hash != manifest.dataset_hash
                or validated_report.candidate_score.model_id != manifest.model_id
                or validated_report.candidate_score.model_revision != manifest.model_revision
            ):
                raise PromotionError("benchmark report metadata does not match the candidate")

            active_manifest = state.candidates.get(state.active_model)
            if active_manifest is not None:
                if (
                    metadata.baseline_model_id != active_manifest.model_id
                    or metadata.baseline_model_revision != active_manifest.model_revision
                ):
                    raise PromotionError("benchmark baseline metadata does not match the active model")
            else:
                if (
                    state.initialized_baseline_model_id is None
                    or state.initialized_baseline_revision is None
                    or metadata.baseline_model_id is None
                    or metadata.baseline_model_revision is None
                    or metadata.baseline_model_id != state.initialized_baseline_model_id
                    or metadata.baseline_model_revision != state.initialized_baseline_revision
                ):
                    raise PromotionError(
                        "benchmark baseline metadata does not match the initialized baseline"
                    )

            updated = state.model_copy(
                update={
                    "active_model": candidate_id,
                    "previous_model": state.active_model,
                    "promoted_at": validated_approval.confirmed_at,
                }
            )
            self._write(updated)
            return updated.model_copy(deep=True)

    def rollback(self) -> RegistryState:
        with self._locked_registry():
            state = self.read()
            if state.active_model is None or state.previous_model is None:
                raise RegistryError("there is no previous model to restore")
            updated = state.model_copy(
                update={
                    "active_model": state.previous_model,
                    "previous_model": state.active_model,
                }
            )
            self._write(updated)
            return updated.model_copy(deep=True)

    def resolve_active_model_name(self, alias: str) -> str:
        if alias != "tuminh-active":
            raise RegistryError(f"unsupported model alias: {alias!r}")
        state = self.read()
        if state.active_model is None or state.active_model not in state.installed_models:
            raise RegistryError("active model is not installed")
        return state.active_model

    @contextmanager
    def _locked_registry(self):
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is not None:
            with open(lock_path, "a+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            return
        if msvcrt is not None:
            with open(lock_path, "a+b") as stream:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            return
        yield

    def _write(self, state: RegistryState) -> None:
        try:
            validated = RegistryState.model_validate(state.model_dump(mode="python"))
            encoded = json.dumps(
                validated.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f"{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as stream:
                    stream.write(encoded)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, self.path)
            finally:
                temporary_path.unlink(missing_ok=True)
        except Exception as error:
            raise RegistryError("registry write failed") from error


def _is_valid_candidate_id(candidate_id: object) -> bool:
    return isinstance(candidate_id, str) and _CANDIDATE_ID.fullmatch(candidate_id) is not None


def _is_valid_model_tag(model_tag: object) -> bool:
    return (
        isinstance(model_tag, str)
        and _MODEL_TAG.fullmatch(model_tag) is not None
        and model_tag not in {".", ".."}
    )
