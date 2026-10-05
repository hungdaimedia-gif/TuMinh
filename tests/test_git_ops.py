from pathlib import Path
import subprocess

import pytest

from agent.models import VerificationResult
from tests.fakes import FakeSelfEvaluator
from tools.command_runner import CommandResult, CommandRunner
from tools.git_ops import GitOperationError, GitOperations


class FakeGitCommandRunner:
    def __init__(self, *, current_branch="main", statuses=None, tracked_diff="tracked diff"):
        self.current_branch = current_branch
        self.branches = {current_branch}
        self.statuses = dict(statuses or {})
        self.tracked_diff = tracked_diff
        self.calls: list[list[str]] = []
        self.staged: list[str] = []
        self.commits: list[list[str]] = []

    def run(self, argv, cwd):
        command = list(argv)
        self.calls.append(command)
        stdout = ""
        if command[:3] == ["git", "branch", "--show-current"]:
            stdout = f"{self.current_branch}\n"
        elif command[:3] == ["git", "branch", "--list"]:
            stdout = f"{command[3]}\n" if command[3] in self.branches else ""
        elif command[:3] == ["git", "status", "--porcelain=v1"]:
            stdout = "".join(f"{status} {path}\0" for path, status in self.statuses.items())
        elif command[:3] == ["git", "switch", "--create"]:
            self.current_branch = command[3]
            self.branches.add(self.current_branch)
        elif command[:2] == ["git", "switch"]:
            self.current_branch = command[2]
        elif command[:4] == ["git", "diff", "--cached", "--name-only"]:
            stdout = "".join(f"{path}\0" for path in self.staged)
        elif command[:2] == ["git", "diff"]:
            if "--check" in command:
                stdout = ""
            elif "--" in command:
                paths = command[command.index("--") + 1 :]
                if any(self.statuses.get(path) == "??" for path in paths):
                    stdout = self.tracked_diff
                else:
                    stdout = self.tracked_diff
        elif command[:2] == ["git", "add"]:
            self.staged = command[command.index("--") + 1 :]
        elif command[:2] == ["git", "commit"]:
            self.commits.append(command)
            stdout = "commit abc1234\n"
        else:
            raise AssertionError(f"Unexpected Git command: {command}")
        return CommandResult(
            argv=command,
            exit_code=0,
            stdout=stdout,
            stderr="",
            timed_out=False,
        )


def test_ensure_task_branch_uses_sanitized_branch_and_preserves_base(tmp_path):
    runner = FakeGitCommandRunner()
    git = GitOperations(tmp_path, runner)

    branch = git.ensure_task_branch("github:issue/4")

    assert branch == "tuminh/task-github-issue-4"
    assert runner.current_branch == branch
    assert "main" in runner.branches
    assert not any(command[:2] == ["git", "commit"] for command in runner.calls)


def test_changed_files_excludes_preexisting_untracked_changes(tmp_path):
    runner = FakeGitCommandRunner(statuses={"user-notes.md": "??"})
    git = GitOperations(tmp_path, runner)
    git.ensure_task_branch("issue-1")
    runner.statuses["agent/new.py"] = "??"

    assert git.changed_files() == ["agent/new.py"]


def test_failed_task_verification_never_commits(tmp_path):
    runner = FakeGitCommandRunner(statuses={"change.py": " M"})
    git = GitOperations(tmp_path, runner)
    git.ensure_task_branch("issue-4")
    failed = VerificationResult(passed=False, commands=[["pytest"]], failures=["failed"])

    with pytest.raises(GitOperationError, match="task verification"):
        git.commit_verified(
            "issue-4",
            "feat: change",
            task_verification=failed,
            self_evaluator=FakeSelfEvaluator(passed=True),
        )

    assert not any(command[:2] == ["git", "add"] for command in runner.calls)
    assert runner.commits == []


def test_failed_self_evaluation_never_commits(tmp_path):
    runner = FakeGitCommandRunner()
    git = GitOperations(tmp_path, runner)
    git.ensure_task_branch("issue-4")
    runner.statuses["agent/change.py"] = " M"

    with pytest.raises(GitOperationError, match="self-evaluation"):
        git.commit_verified(
            "issue-4",
            "feat: change",
            task_verification=VerificationResult(passed=True, commands=[], failures=[]),
            self_evaluator=FakeSelfEvaluator(passed=False),
        )

    assert not any(command[:2] == ["git", "add"] for command in runner.calls)
    assert runner.commits == []


def test_commit_refuses_preexisting_change_that_was_modified_again(tmp_path):
    runner = FakeGitCommandRunner(statuses={"user-file.txt": " M"})
    git = GitOperations(tmp_path, runner)
    git.ensure_task_branch("issue-4")
    runner.statuses["user-file.txt"] = "MM"

    with pytest.raises(GitOperationError, match="pre-existing"):
        git.commit_verified(
            "issue-4",
            "feat: change",
            task_verification=VerificationResult(passed=True, commands=[], failures=[]),
            self_evaluator=FakeSelfEvaluator(passed=True),
        )

    assert runner.commits == []


def test_commit_refuses_ambiguous_staged_changes(tmp_path):
    runner = FakeGitCommandRunner()
    git = GitOperations(tmp_path, runner)
    git.ensure_task_branch("issue-4")
    runner.statuses["change.py"] = "A "

    with pytest.raises(GitOperationError, match="staged"):
        git.commit_verified(
            "issue-4",
            "feat: change",
            task_verification=VerificationResult(passed=True, commands=[], failures=[]),
            self_evaluator=FakeSelfEvaluator(passed=True),
        )

    assert runner.commits == []


def test_commit_stages_only_task_changes_and_returns_commit_output(tmp_path):
    runner = FakeGitCommandRunner(statuses={"user-notes.md": "??"})
    git = GitOperations(tmp_path, runner)
    git.ensure_task_branch("issue-4")
    runner.statuses["src/change.py"] = " M"
    evaluator = FakeSelfEvaluator(passed=True)

    commit = git.commit_verified(
        "issue-4",
        "feat: change",
        task_verification=VerificationResult(passed=True, commands=[["pytest"]], failures=[]),
        self_evaluator=evaluator,
    )

    assert commit == "commit abc1234"
    assert ["git", "add", "--", "src/change.py"] in runner.calls
    assert not any("user-notes.md" in command for command in runner.calls)
    assert len(runner.commits) == 1


def test_diff_text_returns_only_current_task_paths(tmp_path):
    runner = FakeGitCommandRunner(statuses={"base-change.txt": " M"})
    git = GitOperations(tmp_path, runner)
    git.ensure_task_branch("issue-4")
    runner.statuses["task-change.txt"] = " M"

    assert git.diff_text() == "tracked diff"


def test_diff_text_includes_only_new_untracked_task_file(tmp_path):
    runner = FakeGitCommandRunner(statuses={"user-notes.md": "??"})
    git = GitOperations(tmp_path, runner)
    git.ensure_task_branch("issue-4")
    task_file = tmp_path / "new-task-file.txt"
    task_file.write_text("task-only content\n", encoding="utf-8")
    runner.statuses["new-task-file.txt"] = "??"

    diff = git.diff_text()

    assert "task-only content" in diff
    assert "user-notes.md" not in diff


def test_unmerged_status_is_rejected_as_ambiguous(tmp_path):
    runner = FakeGitCommandRunner()
    git = GitOperations(tmp_path, runner)
    git.ensure_task_branch("issue-4")
    runner.statuses["conflicted.py"] = "UU"

    with pytest.raises(GitOperationError, match="ambiguous"):
        git.changed_files()


def test_git_operations_commit_on_dedicated_branch_in_real_temporary_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def run_git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    run_git("init", "-q")
    run_git("config", "user.name", "TuMinh Test")
    run_git("config", "user.email", "tuminh-test@example.invalid")
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    run_git("add", "tracked.txt")
    run_git("commit", "-m", "initial")
    base_branch = run_git("branch", "--show-current")
    base_commit = run_git("rev-parse", "HEAD")
    git = GitOperations(repo, CommandRunner([["git"]], timeout_seconds=15))

    branch = git.ensure_task_branch("integration/1")
    (repo / "agent-change.py").write_text("print('safe')\n", encoding="utf-8")
    result = git.commit_verified(
        "integration/1",
        "feat: integration change",
        task_verification=VerificationResult(passed=True, commands=[], failures=[]),
        self_evaluator=FakeSelfEvaluator(passed=True),
    )

    assert branch == "tuminh/task-integration-1"
    assert result
    assert run_git("branch", "--show-current") == branch
    assert git.changed_files() == []
    run_git("switch", base_branch)
    assert run_git("rev-parse", base_branch) == base_commit
