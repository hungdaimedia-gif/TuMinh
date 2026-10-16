from __future__ import annotations


import json
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


    def create_pull_request(
        self,
        branch: str,
        title: str,
        body: str,
        base: str | None = None,
    ) -> str:
        """Create a pull request for ``branch``.


        ``base`` optionally overrides the PR target branch (passed as
        ``gh pr create --base``). When omitted, ``gh`` targets the
        repository's default branch, which may differ from the branch the
        task branch was cut from.
        """
        if (
            not isinstance(branch, str)
            or not self._BRANCH_PATTERN.fullmatch(branch)
            or branch.startswith("-")
        ):
            raise GitHubOperationError("Pull request branch name is invalid")
        if base is not None and (
            not isinstance(base, str)
            or not self._BRANCH_PATTERN.fullmatch(base)
            or base.startswith("-")
        ):
            raise GitHubOperationError("Pull request base branch name is invalid")
        if not isinstance(title, str) or not title.strip():
            raise GitHubOperationError("Pull request title must not be empty")
        if not isinstance(body, str) or not body.strip():
            raise GitHubOperationError("Pull request body must not be empty")


        existing = self._find_open_pull_request(branch)
        if existing is not None:
            return existing
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
        if base:
            command.extend(["--base", base])
        if self.repository:
            command.extend(["--repo", self.repository])
        result = self._run(command)
        pull_request_url = result.stdout.strip()
        if not pull_request_url:
            raise GitHubOperationError("gh pr create completed without returning a URL")
        return pull_request_url


    def _find_open_pull_request(self, branch: str) -> str | None:
        command = [
            "gh",
            "pr",
            "list",
            "--head",
            branch,
            "--state",
            "open",
            "--json",
            "url",
        ]
        if self.repository:
            command.extend(["--repo", self.repository])
        result = self._run(command)
        try:
            pull_requests = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise GitHubOperationError(
                f"gh pr list returned malformed JSON: {error}"
            ) from error
        if not isinstance(pull_requests, list):
            raise GitHubOperationError("gh pr list must return a JSON array")
        urls: list[str] = []
        for pull_request in pull_requests:
            if not isinstance(pull_request, dict) or not isinstance(
                pull_request.get("url"), str
            ):
                raise GitHubOperationError("gh pr list returned an invalid pull request")
            urls.append(pull_request["url"])
        if len(urls) > 1:
            raise GitHubOperationError(
                f"Multiple open pull requests already exist for branch {branch}"
            )
        return urls[0] if urls else None


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