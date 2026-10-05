from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from agent.models import TaskPlan, VerificationResult
from tools.command_runner import CommandError, CommandRunner


class Verifier(Protocol):
    def verify(
        self,
        plan: TaskPlan,
        changed_files: Sequence[str],
    ) -> VerificationResult: ...


class CommandVerifier:
    def __init__(self, command_runner: CommandRunner, cwd: Path | None = None):
        self.command_runner = command_runner
        self.cwd = Path.cwd() if cwd is None else Path(cwd)

    def verify(
        self,
        plan: TaskPlan,
        changed_files: Sequence[str],
    ) -> VerificationResult:
        del changed_files
        if not plan.verification_commands:
            return VerificationResult(
                passed=False,
                commands=[],
                failures=["No verification commands are configured in the task plan."],
            )

        failures: list[str] = []
        commands = [list(argv) for argv in plan.verification_commands]
        for argv in commands:
            try:
                result = self.command_runner.run(argv, cwd=self.cwd)
            except CommandError as error:
                result = error.result
                if result is None:
                    failures.append(f"{argv!r}: {error}")
                    continue
            if result.timed_out:
                failures.append(f"{argv!r}: command timed out")
            elif result.exit_code != 0:
                failures.append(
                    f"{argv!r}: exit status {result.exit_code}; "
                    f"stdout={result.stdout!r}; stderr={result.stderr!r}"
                )

        return VerificationResult(
            passed=not failures,
            commands=commands,
            failures=failures,
        )
