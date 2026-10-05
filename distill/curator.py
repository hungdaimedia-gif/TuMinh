from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field, ValidationError

from distill.schema import (
    DatasetManifest,
    DistillModel,
    ImportProvenance,
    ObservableEvent,
    TrajectoryExport,
    TrainingMessage,
    TrainingRow,
)


class CuratorError(ValueError):
    """Raised when dataset curation or exact-source purging cannot complete."""


class PurgeResult(DistillModel):
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_files_removed: int = Field(ge=0)
    curated_rows_removed: int = Field(ge=0)
    datasets_updated: int = Field(ge=0)


class Curator:
    _DATASET_VERSION = "v1"
    _CURATION_VERSION = "1.0"
    _SENSITIVE_PATTERNS = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*\S+",
            r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}",
            r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{12,}\b",
            r"\bAKIA[0-9A-Z]{16}\b",
            r"\b[a-z][a-z0-9+.-]*://[^/@\s:]+:[^/@\s]+@",
        )
    )

    def __init__(self, secret_values: Sequence[str]):
        self.secret_values = tuple(
            sorted({value for value in secret_values if value}, key=len, reverse=True)
        )

    def curate(
        self,
        records: Sequence[TrajectoryExport],
        output_path: Path,
    ) -> DatasetManifest:
        validated = self._validate_records(records)
        rows: list[TrainingRow] = []
        provenance_by_hash: dict[str, ImportProvenance] = {}
        counts_by_hash: dict[str, dict[str, int]] = defaultdict(
            lambda: {"configured_secret_values": 0, "credential_patterns": 0}
        )
        for trajectory in validated:
            if (
                trajectory.task_outcome != "succeeded"
                or not trajectory.verification_passed
                or trajectory.eval_passed is not True
            ):
                continue
            source_hash = trajectory.provenance.source_sha256
            existing_provenance = provenance_by_hash.get(source_hash)
            if existing_provenance is not None and existing_provenance != trajectory.provenance:
                raise CuratorError("One source hash has conflicting provenance metadata")
            provenance_by_hash[source_hash] = trajectory.provenance

            messages: list[TrainingMessage] = []
            for event in trajectory.events:
                for message in self._messages_for_event(event):
                    redacted_content, configured_count, pattern_count = self._redact(
                        message.content
                    )
                    counts_by_hash[source_hash]["configured_secret_values"] += configured_count
                    counts_by_hash[source_hash]["credential_patterns"] += pattern_count
                    messages.append(
                        TrainingMessage(
                            role=message.role,
                            content=redacted_content,
                            tool_name=message.tool_name,
                        )
                    )
            if not messages:
                raise CuratorError(
                    f"Eligible trajectory {trajectory.trajectory_id!r} has no observable messages"
                )
            rows.append(
                TrainingRow(
                    task_id=trajectory.task_id,
                    messages=messages,
                    verification_passed=True,
                    eval_passed=True,
                    provenance_id=source_hash,
                )
            )

        source_hashes = sorted(provenance_by_hash)
        aggregate_counts: dict[str, int] = defaultdict(int)
        redaction_counts: dict[str, int] = {}
        for source_hash, source_counts in counts_by_hash.items():
            for category, count in source_counts.items():
                aggregate_counts[category] += count
                redaction_counts[f"source:{source_hash}:{category}"] = count
        redaction_counts.update(aggregate_counts)
        manifest = DatasetManifest(
            dataset_version=self._DATASET_VERSION,
            source_hashes=source_hashes,
            source_provenance=[
                provenance_by_hash[source_hash] for source_hash in source_hashes
            ],
            curation_version=self._CURATION_VERSION,
            redaction_counts=dict(redaction_counts),
            created_at=datetime.now(UTC),
            row_count=len(rows),
        )

        output = Path(output_path)
        manifest_path = output.with_suffix(".manifest.json")
        if output.resolve(strict=False) == manifest_path.resolve(strict=False):
            raise CuratorError("Dataset output and manifest paths must be different")
        rows_content = "".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for row in rows
        )
        _write_atomic_pair(
            {
                output: rows_content,
                manifest_path: manifest.model_dump_json(indent=2) + "\n",
            }
        )
        return manifest

    def purge(
        self,
        source_sha256: str,
        raw_root: Path,
        curated_root: Path,
    ) -> PurgeResult:
        if not re.fullmatch(r"[a-f0-9]{64}", source_sha256):
            raise CuratorError("source_sha256 must be a lowercase SHA-256 value")
        raw_directory = Path(raw_root).resolve(strict=False)
        curated_directory = Path(curated_root).resolve(strict=False)
        raw_path = raw_directory / f"{source_sha256}.jsonl"
        datasets_updated = 0
        curated_rows_removed = 0
        raw_bytes: bytes | None = None
        pending_updates: list[dict[Path, str]] = []

        if raw_path.exists():
            if raw_path.is_symlink() or not raw_path.is_file():
                raise CuratorError("Raw source path is not a regular file")
            try:
                raw_bytes = raw_path.read_bytes()
            except OSError as error:
                raise CuratorError("Could not read requested raw source file") from error
            if hashlib.sha256(raw_bytes).hexdigest() != source_sha256:
                raise CuratorError("Raw source content does not match the requested SHA-256")

        if curated_directory.exists():
            for dataset_path in sorted(curated_directory.rglob("*.jsonl")):
                if dataset_path.is_symlink():
                    raise CuratorError("Refusing to purge through a curated-data symlink")
                manifest_path = dataset_path.with_suffix(".manifest.json")
                if manifest_path.is_symlink():
                    raise CuratorError("Refusing to purge through a curated-manifest symlink")
                if not manifest_path.is_file():
                    continue
                rows, manifest = self._read_dataset_and_manifest(dataset_path, manifest_path)
                if source_sha256 not in manifest.source_hashes:
                    continue
                retained_rows = [
                    row for row in rows if row.provenance_id != source_sha256
                ]
                curated_rows_removed += len(rows) - len(retained_rows)
                retained_hashes = [
                    value for value in manifest.source_hashes if value != source_sha256
                ]
                retained_provenance = [
                    value
                    for value in manifest.source_provenance
                    if value.source_sha256 != source_sha256
                ]
                redaction_counts = _remove_source_redaction_counts(
                    manifest.redaction_counts,
                    source_sha256,
                )
                updated_manifest = DatasetManifest.model_validate(
                    {
                        **manifest.model_dump(mode="python"),
                        "source_hashes": retained_hashes,
                        "source_provenance": retained_provenance,
                        "redaction_counts": redaction_counts,
                        "row_count": len(retained_rows),
                    }
                )
                pending_updates.append(
                    {
                        dataset_path: "".join(
                            json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n"
                            for row in retained_rows
                        ),
                        manifest_path: updated_manifest.model_dump_json(indent=2) + "\n",
                    }
                )
                datasets_updated += 1

        for update in pending_updates:
            _write_atomic_pair(update)

        raw_files_removed = 0
        if raw_bytes is not None:
            try:
                raw_path.unlink()
            except OSError as error:
                raise CuratorError("Could not remove requested raw source file") from error
            raw_files_removed = 1

        return PurgeResult(
            source_sha256=source_sha256,
            raw_files_removed=raw_files_removed,
            curated_rows_removed=curated_rows_removed,
            datasets_updated=datasets_updated,
        )

    def _validate_records(
        self,
        records: Sequence[TrajectoryExport],
    ) -> list[TrajectoryExport]:
        validated: list[TrajectoryExport] = []
        try:
            for record in records:
                payload = record.model_dump(mode="python") if isinstance(record, TrajectoryExport) else record
                validated.append(TrajectoryExport.model_validate(payload))
        except (ValidationError, ValueError) as error:
            raise CuratorError("Curation input contains an invalid trajectory record") from error
        return validated

    @staticmethod
    def _messages_for_event(event: ObservableEvent) -> list[TrainingMessage]:
        message_roles = {
            "user_message": "user",
            "assistant_message": "assistant",
            "tool_message": "tool",
        }
        if event.event_type in message_roles:
            try:
                message = TrainingMessage.model_validate(event.data["message"])
            except (KeyError, ValidationError) as error:
                raise CuratorError(
                    f"Trajectory message at sequence {event.sequence_number} is invalid"
                ) from error
            if message.role != message_roles[event.event_type]:
                raise CuratorError(
                    f"Trajectory message role mismatches sequence {event.sequence_number}"
                )
            return [message]

        if event.event_type == "task":
            values: list[str] = []
            for key in ("title", "description"):
                value = event.data.get(key)
                if isinstance(value, str) and value.strip():
                    values.append(f"{key.title()}: {value.strip()}")
            criteria = event.data.get("acceptance_criteria")
            if isinstance(criteria, list):
                values.extend(
                    f"Acceptance criterion: {value.strip()}"
                    for value in criteria
                    if isinstance(value, str) and value.strip()
                )
            if not values:
                values.append(json.dumps(event.data, ensure_ascii=False, sort_keys=True))
            return [TrainingMessage(role="user", content="\n".join(values))]

        if event.event_type in {"tool_result", "verification"}:
            role = "tool"
        else:
            role = "assistant"
        return [
            TrainingMessage(
                role=role,
                content=json.dumps(event.data, ensure_ascii=False, sort_keys=True),
                tool_name=event.event_type if role == "tool" else None,
            )
        ]

    def _redact(self, content: str) -> tuple[str, int, int]:
        redacted = content
        configured_count = 0
        for secret in self.secret_values:
            occurrences = redacted.count(secret)
            if occurrences:
                configured_count += occurrences
                redacted = redacted.replace(secret, "[REDACTED]")

        pattern_count = 0
        for pattern in self._SENSITIVE_PATTERNS:
            matches = list(pattern.finditer(redacted))
            pattern_count += len(matches)
            redacted = pattern.sub(_redacted_match, redacted)
        return redacted, configured_count, pattern_count

    @staticmethod
    def _read_dataset_and_manifest(
        dataset_path: Path,
        manifest_path: Path,
    ) -> tuple[list[TrainingRow], DatasetManifest]:
        try:
            manifest = DatasetManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            lines = dataset_path.read_text(encoding="utf-8").splitlines()
            rows = [TrainingRow.model_validate_json(line) for line in lines if line]
        except (OSError, UnicodeError, ValidationError, ValueError) as error:
            raise CuratorError("Curated dataset or manifest is invalid") from error
        if len(rows) != manifest.row_count:
            raise CuratorError("Curated dataset row count does not match its manifest")
        return rows, manifest


def _redacted_match(match: re.Match[str]) -> str:
    value = match.group(0)
    if value.lower().startswith("bearer "):
        return "Bearer [REDACTED]"
    if "://" in value and "@" in value:
        scheme = value.split("://", 1)[0]
        return f"{scheme}://[REDACTED]@{value.split('@', 1)[1]}"
    delimiter = max(value.find("="), value.find(":"))
    if delimiter >= 0:
        return value[: delimiter + 1] + "[REDACTED]"
    return "[REDACTED]"


def _remove_source_redaction_counts(
    redaction_counts: Mapping[str, int],
    source_sha256: str,
) -> dict[str, int]:
    retained = {
        key: value
        for key, value in redaction_counts.items()
        if not key.startswith(f"source:{source_sha256}:")
    }
    categories = ("configured_secret_values", "credential_patterns")
    for category in categories:
        retained[category] = sum(
            value
            for key, value in retained.items()
            if key.startswith("source:") and key.endswith(f":{category}")
        )
    return retained


def _write_atomic_pair(contents: Mapping[Path, str]) -> None:
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, content in contents.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                staged.append((temporary_path, destination))
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
        for temporary_path, destination in staged:
            os.replace(temporary_path, destination)
    except OSError as error:
        raise CuratorError("Could not atomically write curated dataset and manifest") from error
    finally:
        for temporary_path, _destination in staged:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
