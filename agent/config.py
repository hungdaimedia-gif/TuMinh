from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping


class SettingsError(ValueError):
    """Raised when environment configuration is invalid."""


TaskSource = Literal["local", "github"]

_DEFAULT_ALLOWED_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("python", "-m", "pytest"),
    ("python", "-m", "evals.run"),
    ("git", "status"),
    ("git", "branch"),
    ("git", "switch"),
    ("git", "diff"),
    ("git", "add"),
    ("git", "commit"),
    ("gh", "issue", "list"),
    ("gh", "pr", "create"),
)


@dataclass(frozen=True, slots=True)
class Settings:
    model_api_key: str | None
    model_base_url: str
    model_name: str
    workspace_root: Path
    tasks_file: Path
    task_state_file: Path
    reflection_file: Path
    task_source: TaskSource
    github_repository: str | None
    max_steps: int
    max_corrections: int
    max_retries: int
    command_timeout_seconds: int
    poll_interval_seconds: int
    max_poll_cycles: int
    allowed_argv_prefixes: tuple[tuple[str, ...], ...]

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        values = os.environ if environ is None else environ

        def positive_int(name: str, default: int) -> int:
            raw = values.get(name, str(default))
            try:
                number = int(raw)
            except (TypeError, ValueError) as error:
                raise SettingsError(f"{name} must be a positive integer") from error
            if number <= 0:
                raise SettingsError(f"{name} must be a positive integer")
            return number

        raw_task_source = values.get("TASK_SOURCE", "local").strip().lower()
        if raw_task_source == "local":
            task_source: TaskSource = "local"
        elif raw_task_source == "github":
            task_source = "github"
        else:
            raise SettingsError("TASK_SOURCE must be 'local' or 'github'")

        workspace_root = Path(values.get("WORKSPACE_ROOT", ".")).expanduser().resolve()
        if not workspace_root.exists() or not workspace_root.is_dir():
            raise SettingsError("WORKSPACE_ROOT must point to an existing directory")

        model_base_url = values.get("MODEL_BASE_URL", "https://api.openai.com/v1").strip()
        model_name = values.get("MODEL_NAME", "gpt-4o-mini").strip()
        if not model_base_url:
            raise SettingsError("MODEL_BASE_URL must not be empty")
        if not model_name:
            raise SettingsError("MODEL_NAME must not be empty")

        tasks_file = Path(values.get("TASKS_FILE", "tasks/tasks.json")).expanduser()
        if not tasks_file.is_absolute():
            tasks_file = workspace_root / tasks_file

        raw_allowed_commands = values.get("ALLOWED_COMMANDS")
        if raw_allowed_commands is None:
            allowed_argv_prefixes = _DEFAULT_ALLOWED_COMMANDS
        else:
            try:
                decoded = json.loads(raw_allowed_commands)
            except json.JSONDecodeError as error:
                raise SettingsError("ALLOWED_COMMANDS must be a JSON array of argv prefixes") from error
            if not isinstance(decoded, list):
                raise SettingsError("ALLOWED_COMMANDS must be a JSON array of argv prefixes")
            if any(
                not isinstance(prefix, list)
                or not prefix
                or any(not isinstance(part, str) or not part for part in prefix)
                for prefix in decoded
            ):
                raise SettingsError("ALLOWED_COMMANDS entries must be non-empty arrays of strings")
            allowed_argv_prefixes = tuple(tuple(prefix) for prefix in decoded)

        api_key = values.get("MODEL_API_KEY", "").strip() or None
        github_repository = values.get("GITHUB_REPOSITORY", "").strip() or None
        return cls(
            model_api_key=api_key,
            model_base_url=model_base_url,
            model_name=model_name,
            workspace_root=workspace_root,
            tasks_file=tasks_file.resolve(),
            task_state_file=workspace_root / "memory" / "task_state.json",
            reflection_file=workspace_root / "memory" / "reflection.json",
            task_source=task_source,
            github_repository=github_repository,
            max_steps=positive_int("MAX_STEPS", 20),
            max_corrections=positive_int("MAX_CORRECTIONS", 2),
            max_retries=positive_int("MAX_RETRIES", 2),
            command_timeout_seconds=positive_int("COMMAND_TIMEOUT_SECONDS", 120),
            poll_interval_seconds=positive_int("POLL_INTERVAL_SECONDS", 30),
            max_poll_cycles=positive_int("MAX_POLL_CYCLES", 20),
            allowed_argv_prefixes=allowed_argv_prefixes,
        )
