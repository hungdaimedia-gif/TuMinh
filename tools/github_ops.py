from __future__ import annotations

import re
from pathlib import Path

from tools.command_runner import CommandError, CommandRunner


class GitHubOperationError(RuntimeError):
    """Raised when a GitHub remote operation fails."""


class GitHubOperations:
    _BRANCH_PATTERN = re.compile(r"[A-Za-z0-9._/-]+\Z")

    def __init__(
        self,
        command_runner: CommandRunner,
        repository: str | None = None,
        cwd: Path | None = None,
    ):
        self.command_runner = command_runner
        self.repository = repository
        self.cwd = Path.cwd() if cwd is None else Path(cwd)

    def create_pull_request(self, branch: str, title: str, body: str) -> str:
        if (
            not isinstance(branch, str)
            or not self._BRANCH_PATTERN.fullmatch(branch)
            or branch.startswith("-")
        ):
            raise GitHubOperationError("Pull request branch name is invalid")
        if not isinstance(title, str) or not title.strip():
            raise GitHubOperationError("Pull request title must not be empty")
        if not isinstance(body, str) or not body.strip():
            raise GitHubOperationError("Pull request body must not be empty")

        self._run(["git", "push", "--set-upstream", "origin", branch])
        command = [
            "gh",
            "pr",
            "create",
            "--title",
            title.strip(),
            "--body",
            body.strip(),
            "--head",
            branch,
        ]
        if self.repository:
            command.extend(["--repo", self.repository])
        result = self._run(command)
        pull_request_url = result.stdout.strip()
        if not pull_request_url:
            raise GitHubOperationError("gh pr create completed without returning a URL")
        return pull_request_url

    def _run(self, argv: list[str]):
        try:
            result = self.command_runner.run(argv, cwd=self.cwd)
        except CommandError as error:
            detail = error.result.stderr.strip() if error.result is not None else str(error)
            raise GitHubOperationError(f"GitHub command failed: {detail}") from error
        if result.timed_out or result.exit_code != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise GitHubOperationError(f"GitHub command failed: {detail}")
        return result
