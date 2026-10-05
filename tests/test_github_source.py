import json

import pytest

from agent.config import Settings
from agent.github_source import GitHubIssueSource, GitHubSourceError
from agent.models import TaskResult
from agent.task_queue import TaskQueue
from tools.command_runner import CommandError, CommandResult
from tools.github_ops import GitHubOperationError, GitHubOperations


class FakeRunner:
    def __init__(self, output="[]", error=None):
        self.output = output
        self.error = error
        self.calls = []

    def run(self, argv, cwd):
        self.calls.append(list(argv))
        if self.error is not None:
            raise self.error
        return CommandResult(
            argv=list(argv),
            exit_code=0,
            stdout=self.output,
            stderr="",
            timed_out=False,
        )


def issue(number=7, title="Add a check", body="Check the requested behavior."):
    return {
        "number": number,
        "title": title,
        "body": body,
        "labels": [{"name": "enhancement"}],
        "state": "OPEN",
    }


def test_github_issue_is_normalized_to_task(tmp_path):
    runner = FakeRunner(json.dumps([issue()]))
    source = GitHubIssueSource(runner, tmp_path / "task_state.json", "owner/repo")

    task = source.next_task()

    assert task is not None
    assert task.id == "github:7"
    assert task.title == "Add a check"
    assert task.description == "Check the requested behavior."
    assert task.source == "github"
    assert task.state == "pending"
    assert runner.calls == [
        [
            "gh",
            "issue",
            "list",
            "--json",
            "number,title,body,labels,state",
            "--state",
            "open",
            "--repo",
            "owner/repo",
        ]
    ]


def test_processed_github_issue_is_skipped(tmp_path):
    state_path = tmp_path / "task_state.json"
    queue = TaskQueue(tmp_path / "unused.json", state_path)
    queue.save_result(
        "github:7",
        TaskResult(
            task_id="github:7",
            status="succeeded",
            summary="done",
        ),
    )
    runner = FakeRunner(json.dumps([issue(7), issue(8, "Next issue")]))

    task = GitHubIssueSource(runner, state_path).next_task()

    assert task is not None
    assert task.id == "github:8"


def test_github_source_returns_none_when_no_open_issues(tmp_path):
    assert GitHubIssueSource(FakeRunner("[]"), tmp_path / "state.json").next_task() is None


def test_closed_issues_are_not_reintroduced_as_succeeded_tasks(tmp_path):
    closed_issue = issue()
    closed_issue["state"] = "CLOSED"
    runner = FakeRunner(json.dumps([closed_issue]))

    assert GitHubIssueSource(runner, tmp_path / "state.json").next_task() is None


def test_malformed_github_json_raises_actionable_error(tmp_path):
    with pytest.raises(GitHubSourceError, match="JSON"):
        GitHubIssueSource(FakeRunner("{broken"), tmp_path / "state.json").next_task()


def test_invalid_or_duplicate_issue_data_is_rejected(tmp_path):
    with pytest.raises(GitHubSourceError, match="issue"):
        GitHubIssueSource(
            FakeRunner(json.dumps([issue(number="not-a-number")])),
            tmp_path / "state.json",
        ).next_task()

    with pytest.raises(GitHubSourceError, match="duplicate"):
        GitHubIssueSource(
            FakeRunner(json.dumps([issue(7), issue(7, "Duplicate")])),
            tmp_path / "state.json",
        ).next_task()


def test_missing_gh_and_authentication_errors_are_visible(tmp_path):
    unavailable = CommandError("gh executable was not found")
    with pytest.raises(GitHubSourceError, match="gh"):
        GitHubIssueSource(
            FakeRunner(error=unavailable),
            tmp_path / "state.json",
        ).next_task()

    auth_error = CommandError(
        "GitHub authentication failed",
        result=CommandResult(
            argv=["gh", "issue", "list"],
            exit_code=4,
            stdout="",
            stderr="not logged in",
            timed_out=False,
        ),
    )
    with pytest.raises(GitHubSourceError, match="not logged in"):
        GitHubIssueSource(
            FakeRunner(error=auth_error),
            tmp_path / "state.json",
        ).next_task()


def test_pull_request_pushes_only_after_caller_requests_and_returns_url(tmp_path):
    runner = FakeRunner("https://github.com/owner/repo/pull/9\n")
    settings = Settings.from_env(
        {
            "WORKSPACE_ROOT": str(tmp_path),
            "GITHUB_REPOSITORY": "owner/repo",
        }
    )

    url = GitHubOperations(runner, settings.github_repository).create_pull_request(
        "tuminh/task-issue-7",
        "Add a check",
        "Task complete and verified.",
    )

    assert url == "https://github.com/owner/repo/pull/9"
    assert runner.calls == [
        ["git", "push", "--set-upstream", "origin", "tuminh/task-issue-7"],
        [
            "gh",
            "pr",
            "create",
            "--title",
            "Add a check",
            "--body",
            "Task complete and verified.",
            "--head",
            "tuminh/task-issue-7",
            "--repo",
            "owner/repo",
        ],
    ]


def test_pull_request_failure_is_reported(tmp_path):
    runner = FakeRunner(
        error=CommandError(
            "push failed",
            result=CommandResult(
                argv=["git", "push"],
                exit_code=1,
                stdout="",
                stderr="remote unavailable",
                timed_out=False,
            ),
        )
    )

    with pytest.raises(GitHubOperationError, match="remote unavailable"):
        GitHubOperations(runner).create_pull_request(
            "tuminh/task-issue-7",
            "Title",
            "Body",
        )


def test_pull_request_creation_failure_after_push_is_visible(tmp_path):
    class FailCreateRunner:
        def __init__(self):
            self.calls = []

        def run(self, argv, cwd):
            command = list(argv)
            self.calls.append(command)
            if command[:3] == ["gh", "pr", "create"]:
                raise CommandError(
                    "pull request failed",
                    result=CommandResult(
                        argv=command,
                        exit_code=1,
                        stdout="",
                        stderr="permission denied",
                        timed_out=False,
                    ),
                )
            return CommandResult(
                argv=command,
                exit_code=0,
                stdout="",
                stderr="",
                timed_out=False,
            )

    runner = FailCreateRunner()
    with pytest.raises(GitHubOperationError, match="permission denied"):
        GitHubOperations(runner).create_pull_request(
            "tuminh/task-issue-7",
            "Title",
            "Body",
        )
    assert runner.calls[0][:2] == ["git", "push"]
