from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


class CommandError(RuntimeError):
    """Raised when a command is rejected or does not complete successfully."""

    def __init__(self, message: str, result: CommandResult | None = None):
        super().__init__(message)
        self.result = result


class CommandRunner:
    def __init__(
        self,
        allowed_argv_prefixes: Sequence[Sequence[str]],
        timeout_seconds: int,
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        prefixes = [tuple(prefix) for prefix in allowed_argv_prefixes]
        if any(not prefix or any(not part for part in prefix) for prefix in prefixes):
            raise ValueError("Allowed argv prefixes must be non-empty")
        self.allowed_argv_prefixes = tuple(prefixes)
        self.timeout_seconds = timeout_seconds

    def run(self, argv: Sequence[str], cwd: Path) -> CommandResult:
        command = list(argv)
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise CommandError("Command argv must be a non-empty sequence of strings")
        if any("\x00" in part for part in command):
            raise CommandError("Command argv cannot contain NUL characters")
        if not any(
            len(command) >= len(prefix) and tuple(command[: len(prefix)]) == prefix
            for prefix in self.allowed_argv_prefixes
        ):
            raise CommandError(f"Command is not allowlisted: {command!r}")

        try:
            completed = subprocess.run(
                command,
                shell=False,
                cwd=Path(cwd),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            result = CommandResult(
                argv=command,
                exit_code=None,
                stdout=_to_text(error.stdout),
                stderr=_to_text(error.stderr),
                timed_out=True,
            )
            raise CommandError(
                f"Command timed out after {self.timeout_seconds} seconds: {command!r}",
                result=result,
            ) from error
        except OSError as error:
            raise CommandError(f"Could not execute command {command!r}: {error}") from error

        result = CommandResult(
            argv=command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            timed_out=False,
        )
        if completed.returncode != 0:
            raise CommandError(
                f"Command exited with status {completed.returncode}: {command!r}",
                result=result,
            )
        return result


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
