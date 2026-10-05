from __future__ import annotations

from pathlib import Path
from typing import Sequence

from agent.models import VerificationResult
from tools.command_runner import CommandError, CommandRunner


_SELF_TEST_COMMANDS = (
    ("python", "-m", "pytest", "-q"),
    ("python", "-m", "evals.run"),
)
_SELF_MODIFIED_PATHS = (
    "agent",
    "tools",
    "tests",
    "evals",
    "skills/approved",
    "main.py",
    "pyproject.toml",
    ".env.example",
)


class SelfEvaluator:
    def __init__(self, agent_root: Path, command_runner: CommandRunner):
        self.agent_root = Path(agent_root).resolve()
        self.command_runner = command_runner

    def verify_before_commit(self, changed_files: Sequence[str]) -> VerificationResult:
        if not any(_is_self_modified(path) for path in changed_files):
            return VerificationResult(passed=True, commands=[], failures=[])

        failures: list[str] = []
        commands = [list(command) for command in _SELF_TEST_COMMANDS]
        for argv in commands:
            try:
                result = self.command_runner.run(argv, cwd=self.agent_root)
            except CommandError as error:
                result = error.result
                if result is None:
                    failures.append(f"{argv!r}: {error}")
                    continue
            if result.timed_out:
                failures.append(f"{argv!r}: self-evaluation command timed out")
            elif result.exit_code != 0:
                failures.append(
                    f"{argv!r}: exit status {result.exit_code}; "
                    f"stderr={result.stderr!r}"
                )

        return VerificationResult(
            passed=not failures,
            commands=commands,
            failures=failures,
        )


def _is_self_modified(path: str) -> bool:
    normalized = path.replace("\\", "/").removeprefix("./").strip("/")
    return any(normalized == root or normalized.startswith(f"{root}/") for root in _SELF_MODIFIED_PATHS)
