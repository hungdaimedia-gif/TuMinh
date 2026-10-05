from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class TrajectoryError(ValueError):
    """Raised when trajectory data is unsafe, invalid, or cannot be persisted."""


class TrajectoryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
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
    ]
    timestamp: datetime
    data: dict[str, object]


class TrajectoryJournal:
    MAX_EVENT_BYTES = 64 * 1024
    MAX_DATA_BYTES = 48 * 1024
    MAX_PATCH_BYTES = 32 * 1024
    MAX_STRING_BYTES = 12 * 1024

    _TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
    _PROHIBITED_KEY_PARTS = (
        "apikey",
        "credential",
        "password",
        "reasoning",
        "secret",
        "token",
    )
    _PROHIBITED_KEYS = {
        "analysis",
        "chainofthought",
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
    _SENSITIVE_VALUE_PATTERNS = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*(?!\[REDACTED\]|\*+)[^\s,;]+",
            r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}",
            r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{12,}\b",
            r"\bAKIA[0-9A-Z]{16}\b",
            r"\bsk-[A-Za-z0-9]{16,}\b",
            r"\b(?:database|db|connection|redis|mongo|postgres|mysql)[_-]?(?:url|uri)\s*[:=]\s*[^\s,;]+",
        )
    )
    _URL_USERINFO_PATTERN = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^/@\s]+@")
    _ALLOWED_EVENT_TYPES = {
        "task",
        "plan",
        "decision",
        "tool_call",
        "tool_result",
        "patch",
        "verification",
        "reflection",
    }

    def __init__(self, root: Path, secret_values: Sequence[str] = ()):
        self.root = Path(root).expanduser().resolve(strict=False)
        self.secret_values = tuple(
            sorted({value for value in secret_values if value}, key=len, reverse=True)
        )

    def path_for_task(self, task_id: str) -> Path:
        if not isinstance(task_id, str) or not self._TASK_ID_PATTERN.fullmatch(task_id):
            raise TrajectoryError("Unsafe trajectory task id")
        if ".." in task_id:
            raise TrajectoryError("Unsafe trajectory task id")

        safe_id = task_id.replace(":", "-")
        if safe_id != task_id:
            suffix = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8]
            safe_id = f"{safe_id}-{suffix}"
        path = self.root / f"{safe_id}.jsonl"
        try:
            path.resolve(strict=False).relative_to(self.root)
        except (OSError, ValueError) as error:
            raise TrajectoryError("Trajectory path escapes its journal root") from error
        return path

    def append(
        self,
        task_id: str,
        event_type: str,
        data: Mapping[str, object],
    ) -> None:
        if event_type not in self._ALLOWED_EVENT_TYPES:
            raise TrajectoryError(f"Unknown trajectory event type: {event_type!r}")
        if not isinstance(data, Mapping):
            raise TrajectoryError("Trajectory event data must be a JSON object")
        if event_type == "patch" and (set(data) != {"diff"} or not isinstance(data["diff"], str)):
            raise TrajectoryError("Patch events may contain only a string diff")
        if event_type == "patch" and len(data["diff"].encode("utf-8")) > self.MAX_PATCH_BYTES:
            raise TrajectoryError("Patch event exceeds the configured size limit")
        self._reject_prohibited_fields(data)

        sanitized_data = self._sanitize_value(data)
        if not isinstance(sanitized_data, dict):
            raise TrajectoryError("Trajectory event data must be a JSON object")
        if event_type == "patch":
            patch = sanitized_data["diff"]
            if len(patch.encode("utf-8")) > self.MAX_PATCH_BYTES:
                raise TrajectoryError("Patch event exceeds the configured size limit")

        path = self.path_for_task(task_id)
        self._ensure_root()
        events = self._read_path(path, task_id)
        event = TrajectoryEvent(
            schema_version=1,
            task_id=task_id,
            sequence_number=len(events) + 1,
            event_type=event_type,
            timestamp=datetime.now(UTC),
            data=sanitized_data,
        )
        serialized = event.model_dump_json()
        encoded = serialized.encode("utf-8")
        data_bytes = json.dumps(
            sanitized_data,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if len(data_bytes) > self.MAX_DATA_BYTES or len(encoded) > self.MAX_EVENT_BYTES:
            raise TrajectoryError("Trajectory event exceeds the configured size limit")

        try:
            with path.open("a", encoding="utf-8", newline="\n") as journal_file:
                journal_file.write(serialized + "\n")
                journal_file.flush()
                os.fsync(journal_file.fileno())
        except OSError as error:
            raise TrajectoryError(f"Could not append trajectory event: {error}") from error

    def read(self, task_id: str) -> list[TrajectoryEvent]:
        return self._read_path(self.path_for_task(task_id), task_id)

    def _read_path(self, path: Path, task_id: str) -> list[TrajectoryEvent]:
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            raise TrajectoryError(f"Could not read trajectory file {path}: {error}") from error
        events: list[TrajectoryEvent] = []
        for line_number, line in enumerate(lines, start=1):
            try:
                event = TrajectoryEvent.model_validate_json(line)
            except (ValidationError, ValueError) as error:
                raise TrajectoryError(
                    f"Malformed trajectory record at line {line_number}: {error}"
                ) from error
            if any(secret in line for secret in self.secret_values) or any(
                pattern.search(line) for pattern in self._SENSITIVE_VALUE_PATTERNS
            ):
                raise TrajectoryError(
                    f"Trajectory record at line {line_number} contains a sensitive value"
                )
            self._reject_prohibited_fields(event.data)
            if event.task_id != task_id:
                raise TrajectoryError(
                    f"Trajectory record at line {line_number} has a mismatched task id"
                )
            if event.sequence_number != line_number:
                raise TrajectoryError(
                    f"Trajectory sequence mismatch at line {line_number}"
                )
            events.append(event)
        return events

    def _ensure_root(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if self.root.resolve(strict=True) != self.root:
                raise TrajectoryError("Trajectory root resolves outside its configured path")
        except OSError as error:
            raise TrajectoryError(f"Could not create trajectory directory: {error}") from error

    def _reject_prohibited_fields(self, value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                if not isinstance(key, str):
                    raise TrajectoryError("Trajectory object keys must be strings")
                normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
                if (
                    normalized_key in self._PROHIBITED_KEYS
                    or normalized_key.startswith(
                        (
                            "analysis",
                            "chainofthought",
                            "thinking",
                            "thought",
                            "systemprompt",
                            "developerprompt",
                        )
                    )
                    or normalized_key.startswith("environment")
                    or normalized_key == "env"
                    or normalized_key.startswith("envvar")
                    or any(part in normalized_key for part in self._PROHIBITED_KEY_PARTS)
                ):
                    raise TrajectoryError(
                        f"Trajectory contains a prohibited field: {key}"
                    )
                self._reject_prohibited_fields(nested_value)
        elif isinstance(value, list):
            for item in value:
                self._reject_prohibited_fields(item)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, float) and not (float("-inf") < value < float("inf")):
                raise TrajectoryError("Trajectory data cannot contain non-finite numbers")
        else:
            raise TrajectoryError(
                f"Trajectory data contains a non-JSON value: {type(value).__name__}"
            )

    def _sanitize_value(self, value: object) -> object:
        if isinstance(value, str):
            sanitized = value
            for secret in self.secret_values:
                sanitized = sanitized.replace(secret, "[REDACTED]")
            sanitized = self._URL_USERINFO_PATTERN.sub(
                r"\1[REDACTED]@",
                sanitized,
            )
            for pattern in self._SENSITIVE_VALUE_PATTERNS:
                sanitized = pattern.sub(lambda match: _redacted_match(match.group(0)), sanitized)
            encoded = sanitized.encode("utf-8")
            if len(encoded) > self.MAX_STRING_BYTES:
                marker = "\n[truncated]"
                limit = self.MAX_STRING_BYTES - len(marker.encode("utf-8"))
                sanitized = encoded[:limit].decode("utf-8", errors="ignore") + marker
            return sanitized
        if isinstance(value, Mapping):
            return {key: self._sanitize_value(nested) for key, nested in value.items()}
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        return value


def _redacted_match(value: str) -> str:
    key_separator = max(value.find("="), value.find(":"))
    if key_separator >= 0:
        return value[: key_separator + 1] + "[REDACTED]"
    if value.lower().startswith("bearer "):
        return "Bearer [REDACTED]"
    return "[REDACTED]"
