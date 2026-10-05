from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from agent.trajectory import TrajectoryError, TrajectoryJournal
from distill.schema import (
    ExternalExportHeader,
    ImportProvenance,
    ObservableEvent,
    RuntimeExportHeader,
    TrainingMessage,
    TrajectoryExport,
)


class DistillImportError(ValueError):
    """Raised when an explicitly supplied trajectory export is invalid."""


class RuntimeExporter:
    def __init__(self, journal_root: Path, secret_values: Sequence[str] = ()):
        self.journal_root = Path(journal_root)
        self.secret_values = tuple(secret_values)

    def export(self, task_id: str, output_path: Path) -> Path:
        journal = TrajectoryJournal(self.journal_root, secret_values=self.secret_values)
        try:
            source_path = journal.path_for_task(task_id)
            source_bytes = source_path.read_bytes()
            events = journal.read(task_id)
            if source_path.read_bytes() != source_bytes:
                raise DistillImportError(
                    "Runtime trajectory changed while it was being exported"
                )
        except (TrajectoryError, OSError) as error:
            raise DistillImportError(
                f"Could not read validated runtime trajectory for task {task_id!r}"
            ) from error
        if not events or not any(event.event_type == "task" for event in events):
            raise DistillImportError("Runtime trajectory must include a task event")

        reflections = [event for event in events if event.event_type == "reflection"]
        verifications = [event for event in events if event.event_type == "verification"]
        if not reflections or not verifications:
            raise DistillImportError(
                "Runtime trajectory must include reflection and verification events"
            )
        task_outcome = reflections[-1].data.get("outcome")
        if not isinstance(task_outcome, str) or task_outcome not in {"succeeded", "failed"}:
            raise DistillImportError("Runtime trajectory reflection has an invalid outcome")
        verification_data = verifications[-1].data.get("result", verifications[-1].data)
        verification_passed = (
            verification_data.get("passed")
            if isinstance(verification_data, Mapping)
            else None
        )
        if not isinstance(verification_passed, bool):
            raise DistillImportError("Runtime trajectory has no valid verification result")
        eval_passed = _runtime_eval_result(events)
        imported_at = datetime.now(UTC)
        provenance = ImportProvenance(
            source_name="tuminh-runtime",
            imported_at=imported_at,
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        )
        export = TrajectoryExport(
            schema_version=1,
            trajectory_id=f"runtime-{task_id}",
            task_id=task_id,
            source_type="runtime",
            provenance=provenance,
            task_outcome=task_outcome,
            verification_passed=verification_passed,
            eval_passed=eval_passed,
            events=[
                ObservableEvent.model_validate(event.model_dump(mode="json"))
                for event in events
            ],
        )
        output = Path(output_path)
        if output.resolve(strict=False) == source_path.resolve(strict=False):
            raise DistillImportError("Runtime export destination cannot overwrite its source")
        write_trajectory_exports([export], output)
        return output


class ExternalImporter:
    _MAX_SOURCE_FILE_BYTES = 64 * 1024 * 1024
    _MAX_EVENTS_PER_TRAJECTORY = 1_000
    _MAX_EVENT_BYTES = 64 * 1024

    _ALLOWED_EVENT_TYPES = {
        "task",
        "user_message",
        "assistant_message",
        "tool_message",
        "verification",
    }
    _SENSITIVE_TEXT_PATTERNS = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*\S+",
            r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}",
            r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{12,}\b",
            r"\bAKIA[0-9A-Z]{16}\b",
            r"\b[a-z][a-z0-9+.-]*://[^/@\s:]+:[^/@\s]+@",
        )
    )

    def __init__(self, secret_values: Sequence[str] = ()):
        self.secret_values = tuple(value for value in secret_values if value)

    def import_file(
        self,
        path: Path,
        *,
        source_name: str,
        license_or_authorization: str,
        authorized_by: str,
        source_model_id: str | None = None,
        source_model_revision: str | None = None,
    ) -> list[TrajectoryExport]:
        source_path = Path(path)
        if not source_name.strip():
            raise DistillImportError(f"{source_path}: source_name is required")
        if not license_or_authorization.strip():
            raise DistillImportError(f"{source_path}: license_or_authorization is required")
        if not authorized_by.strip():
            raise DistillImportError(f"{source_path}: authorized_by is required")

        try:
            source_stat = source_path.stat()
        except OSError as error:
            raise DistillImportError(f"{source_path}: Could not access the supplied external export") from error
        if source_stat.st_size > self._MAX_SOURCE_FILE_BYTES:
            raise DistillImportError(f"{source_path}: External export exceeds size limit")

        try:
            source_bytes = source_path.read_bytes()
        except OSError as error:
            raise DistillImportError(f"{source_path}: Could not read the supplied external export") from error
        if len(source_bytes) > self._MAX_SOURCE_FILE_BYTES:
            raise DistillImportError(f"{source_path}: External export exceeds size limit")
        if not source_bytes:
            raise DistillImportError(f"{source_path}: External export file is empty")
        try:
            content = source_bytes.decode("utf-8")
        except UnicodeError as error:
            raise DistillImportError(f"{source_path}: External export is not valid UTF-8") from error

        try:
            blocks = _split_export_blocks(
                content,
                max_events_per_trajectory=self._MAX_EVENTS_PER_TRAJECTORY,
                max_event_bytes=self._MAX_EVENT_BYTES,
            )
        except DistillImportError as error:
            raise DistillImportError(f"{source_path}: {error}") from error

        source_hash = hashlib.sha256(source_bytes).hexdigest()
        try:
            provenance = ImportProvenance(
                source_name=source_name,
                license_or_authorization=license_or_authorization,
                authorized_by=authorized_by,
                source_model_id=source_model_id,
                source_model_revision=source_model_revision,
                imported_at=datetime.now(UTC),
                source_sha256=source_hash,
            )
        except ValidationError as error:
            raise DistillImportError(
                f"{source_path}: External import provenance is invalid"
            ) from error

        trajectories: list[TrajectoryExport] = []
        seen_trajectory_ids: set[str] = set()
        for header_line, event_lines in blocks:
            try:
                header = ExternalExportHeader.model_validate_json(header_line)
                events = [ObservableEvent.model_validate_json(line) for line in event_lines]
                self._validate_external_events(events)
                trajectory = TrajectoryExport(
                    schema_version=header.schema_version,
                    trajectory_id=header.trajectory_id,
                    task_id=header.task_id,
                    source_type="external",
                    provenance=provenance,
                    task_outcome=header.task_outcome,
                    verification_passed=header.verification_passed,
                    eval_passed=header.eval_passed,
                    events=events,
                )
            except DistillImportError as error:
                raise DistillImportError(f"{source_path}: {error}") from error
            except (ValidationError, ValueError) as error:
                raise DistillImportError(
                    f"{source_path}: External export contains an invalid trajectory schema or unsafe field"
                ) from error
            if trajectory.trajectory_id in seen_trajectory_ids:
                raise DistillImportError(f"{source_path}: External export contains a duplicate trajectory id")
            seen_trajectory_ids.add(trajectory.trajectory_id)
            trajectories.append(trajectory)
        return trajectories

    def _validate_external_events(self, events: Sequence[ObservableEvent]) -> None:
        found_message = False
        for event in events:
            if event.event_type not in self._ALLOWED_EVENT_TYPES:
                raise DistillImportError(
                    f"Unsupported external event type at sequence {event.sequence_number}"
                )
            _reject_sensitive_values(event.data, self.secret_values, self._SENSITIVE_TEXT_PATTERNS)
            expected_role = {
                "user_message": "user",
                "assistant_message": "assistant",
                "tool_message": "tool",
            }.get(event.event_type)
            if expected_role is not None:
                if set(event.data) != {"message"}:
                    raise DistillImportError(
                        f"External message event {event.sequence_number} has unknown fields"
                    )
                try:
                    message = TrainingMessage.model_validate(event.data["message"])
                except ValidationError as error:
                    raise DistillImportError(
                        f"External message event {event.sequence_number} has an invalid role or content"
                    ) from error
                if message.role != expected_role:
                    raise DistillImportError(
                        f"External message event {event.sequence_number} has a mismatched role"
                    )
                found_message = True
        if not found_message:
            raise DistillImportError("External trajectory contains no user/assistant/tool messages")


def _split_export_blocks(
    content: str,
    *,
    max_events_per_trajectory: int,
    max_event_bytes: int,
) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    current_header: str | None = None
    current_events: list[str] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            raise DistillImportError(f"External export contains an empty line at {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise DistillImportError(
                f"External export contains malformed JSON at line {line_number}"
            ) from error
        if isinstance(record, dict) and record.get("record_type") == "header":
            if current_header is not None:
                if not current_events:
                    raise DistillImportError("External trajectory header has no events")
                if len(current_events) > max_events_per_trajectory:
                    raise DistillImportError(
                        "External trajectory exceeds event count limit"
                    )
                blocks.append((current_header, current_events))
            current_header = line
            current_events = []
        elif current_header is None:
            raise DistillImportError(
                f"External export must start with a trajectory header at line {line_number}"
            )
        else:
            if len(current_events) >= max_events_per_trajectory:
                raise DistillImportError("External trajectory exceeds event count limit")
            encoded_line = line.encode("utf-8")
            if len(encoded_line) > max_event_bytes:
                raise DistillImportError(
                    "External event exceeds encoded size limit"
                )
            current_events.append(line)
    if current_header is None or not current_events:
        raise DistillImportError("External export must contain a header and at least one event")
    if len(current_events) > max_events_per_trajectory:
        raise DistillImportError("External trajectory exceeds event count limit")
    blocks.append((current_header, current_events))
    return blocks


def _runtime_eval_result(events: Sequence[object]) -> bool | None:
    for event in reversed(events):
        if event.event_type != "verification":
            continue
        data = event.data.get("result", event.data)
        if isinstance(data, Mapping) and isinstance(data.get("eval_passed"), bool):
            return data["eval_passed"]
    return None


def write_trajectory_exports(
    trajectories: Sequence[TrajectoryExport],
    output_path: Path,
) -> Path:
    if not trajectories:
        raise DistillImportError("At least one validated trajectory is required")
    records: list[str] = []
    for trajectory in trajectories:
        header = RuntimeExportHeader(
            record_type="header",
            schema_version=trajectory.schema_version,
            trajectory_id=trajectory.trajectory_id,
            task_id=trajectory.task_id,
            source_type=trajectory.source_type,
            provenance=trajectory.provenance,
            task_outcome=trajectory.task_outcome,
            verification_passed=trajectory.verification_passed,
            eval_passed=trajectory.eval_passed,
        )
        records.extend(
            [
                header.model_dump_json(),
                *(event.model_dump_json() for event in trajectory.events),
            ]
        )
    output = Path(output_path)
    _write_atomic(output, "\n".join(records) + "\n")
    return output


def _reject_sensitive_values(
    value: object,
    secret_values: Sequence[str],
    patterns: Sequence[re.Pattern[str]],
) -> None:
    if isinstance(value, str):
        if any(secret in value for secret in secret_values) or any(
            pattern.search(value) for pattern in patterns
        ):
            raise DistillImportError("External export contains a credential-like value")
    elif isinstance(value, Mapping):
        for nested in value.values():
            _reject_sensitive_values(nested, secret_values, patterns)
    elif isinstance(value, list):
        for nested in value:
            _reject_sensitive_values(nested, secret_values, patterns)


def _write_atomic(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        raise DistillImportError("Could not atomically write trajectory export") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
