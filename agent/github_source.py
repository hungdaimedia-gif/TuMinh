from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from agent.models import Task
from agent.task_queue import TaskQueue
from tools.command_runner import CommandError, CommandRunner


class GitHubSourceError(RuntimeError):
    """Raised when GitHub Issues cannot be listed or normalized."""


class GitHubIssueSource:
    def __init__(
        self,
        command_runner: CommandRunner,
        state_path: Path,
        repository: str | None = None,
        cwd: Path | None = None,
    ):
        self.command_runner = command_runner
        self.repository = repository
        self.cwd = Path.cwd() if cwd is None else Path(cwd)
        self.state_queue = TaskQueue(Path("tasks") / "tasks.json", state_path)

    def next_task(self) -> Task | None:
        command = [
            "gh",
            "issue",
            "list",
            "--json",
            "number,title,body,labels,state",
            "--state",
            "open",
        ]
        if self.repository:
            command.extend(["--repo", self.repository])
        try:
            result = self.command_runner.run(command, cwd=self.cwd)
        except CommandError as error:
            detail = error.result.stderr.strip() if error.result is not None else str(error)
            raise GitHubSourceError(f"Could not list GitHub Issues with gh: {detail}") from error
        if result.timed_out or result.exit_code != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise GitHubSourceError(f"Could not list GitHub Issues: {detail}")
        try:
            raw_issues = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise GitHubSourceError(f"GitHub returned malformed issue JSON: {error}") from error
        if not isinstance(raw_issues, list):
            raise GitHubSourceError("GitHub issue response must be a JSON array")

        seen_numbers: set[int] = set()
        normalized_issues: list[Task] = []
        for index, raw_issue in enumerate(raw_issues):
            if isinstance(raw_issue, dict):
                issue_number = raw_issue.get("number")
                if isinstance(issue_number, bool) or not isinstance(issue_number, int) or issue_number <= 0:
                    raise GitHubSourceError(
                        f"GitHub issue at result index {index} has an invalid number"
                    )
                if issue_number in seen_numbers:
                    raise GitHubSourceError(
                        f"GitHub returned duplicate issue number {issue_number}"
                    )
                seen_numbers.add(issue_number)
            task = self._normalize_issue(raw_issue, index)
            if task is not None:
                normalized_issues.append(task)

        for task in normalized_issues:
            if task.state in {"succeeded", "failed"}:
                continue
            if not self.state_queue.is_processed(task.id):
                return task
        return None

    @staticmethod
    def _normalize_issue(raw_issue: object, index: int) -> Task | None:
        if not isinstance(raw_issue, dict):
            raise GitHubSourceError(f"Invalid GitHub issue at result index {index}")
        number = raw_issue.get("number")
        title = raw_issue.get("title")
        body = raw_issue.get("body")
        labels = raw_issue.get("labels", [])
        state = raw_issue.get("state", "OPEN")
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise GitHubSourceError(f"GitHub issue at result index {index} has an invalid number")
        if not isinstance(title, str) or not title.strip():
            raise GitHubSourceError(f"GitHub issue #{number} has an invalid title")
        if body is not None and not isinstance(body, str):
            raise GitHubSourceError(f"GitHub issue #{number} has an invalid body")
        if not isinstance(labels, list) or any(
            not isinstance(label, dict) or not isinstance(label.get("name"), str)
            for label in labels
        ):
            raise GitHubSourceError(f"GitHub issue #{number} has invalid labels")
        if not isinstance(state, str):
            raise GitHubSourceError(f"GitHub issue #{number} has an invalid state")

        if state.upper() != "OPEN":
            return None
        try:
            return Task(
                id=f"github:{number}",
                title=title,
                description=(body or "").strip() or title,
                acceptance_criteria=[],
                source="github",
                state="pending",
            )
        except ValidationError as error:
            raise GitHubSourceError(f"Invalid GitHub issue #{number}: {error}") from error
