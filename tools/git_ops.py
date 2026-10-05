from __future__ import annotations

import difflib
import re
from pathlib import Path, PurePosixPath
from typing import Protocol, Sequence

from agent.models import VerificationResult
from tools.command_runner import CommandError, CommandRunner


class GitOperationError(RuntimeError):
    """Raised when a Git operation violates a branch or commit safety gate."""


class SelfEvaluator(Protocol):
    def verify_before_commit(self, changed_files: Sequence[str]) -> VerificationResult: ...


class GitOperations:
    _TASK_ID_PATTERN = re.compile(r"[^a-z0-9]+")

    def __init__(self, workspace: Path, command_runner: CommandRunner):
        self.workspace = Path(workspace).resolve()
        self.command_runner = command_runner
        self._task_branch: str | None = None
        self._base_branch: str | None = None
        self._baseline_status: dict[str, str] = {}

    def ensure_task_branch(self, task_id: str) -> str:
        target_branch = self._branch_for_task(task_id)
        if self._task_branch == target_branch:
            self._assert_current_task_branch()
            return target_branch
        if self._task_branch is not None and self.changed_files():
            raise GitOperationError("Cannot switch tasks with uncommitted changes on the current task branch")

        current_branch = self._current_branch()
        if not current_branch:
            raise GitOperationError("Cannot create a task branch from a detached HEAD")
        if self._base_branch is None:
            if current_branch.startswith("tuminh/task-"):
                raise GitOperationError(
                    "Cannot establish a safe base branch from an existing TuMinh task branch"
                )
            self._base_branch = current_branch
        elif current_branch != self._base_branch:
            self._run(["git", "switch", self._base_branch])
            current_branch = self._current_branch()
            if current_branch != self._base_branch:
                raise GitOperationError("Could not return to the base branch")

        baseline = self._read_status()
        branch_listing = self._run(["git", "branch", "--list", target_branch])
        if branch_listing.stdout.strip():
            raise GitOperationError(
                f"Task branch already exists but is not active: {target_branch}"
            )
        self._run(["git", "switch", "--create", target_branch])
        if self._current_branch() != target_branch:
            raise GitOperationError(f"Could not switch to task branch {target_branch}")

        self._task_branch = target_branch
        self._baseline_status = baseline
        return target_branch

    def changed_files(self) -> list[str]:
        self._assert_current_task_branch()
        return sorted(self._task_changes())

    def diff_text(self) -> str:
        self._assert_current_task_branch()
        status = self._task_changes()
        tracked_paths = sorted(path for path, state in status.items() if state != "??")
        diff_parts: list[str] = []
        if tracked_paths:
            result = self._run(["git", "diff", "HEAD", "--", *tracked_paths])
            if result.stdout:
                diff_parts.append(result.stdout)

        for relative_path in sorted(path for path, state in status.items() if state == "??"):
            target = self._workspace_file(relative_path)
            if target.is_symlink():
                raise GitOperationError(f"Refusing to read a symlink diff: {relative_path}")
            try:
                raw_content = target.read_bytes()
            except OSError as error:
                raise GitOperationError(
                    f"Could not read untracked change {relative_path}: {error}"
                ) from error
            try:
                content = raw_content.decode("utf-8")
            except UnicodeDecodeError:
                diff_parts.append(f"Binary file added: {relative_path}\n")
                continue
            lines = content.splitlines(keepends=True)
            if content and (not lines or not lines[-1].endswith(("\n", "\r"))):
                lines[-1] += "\n"
            diff_parts.extend(
                difflib.unified_diff(
                    [],
                    lines,
                    fromfile="/dev/null",
                    tofile=f"b/{relative_path}",
                )
            )
        return "".join(diff_parts)

    def commit_verified(
        self,
        task_id: str,
        message: str,
        *,
        task_verification: VerificationResult,
        self_evaluator: SelfEvaluator,
    ) -> str:
        expected_branch = self._branch_for_task(task_id)
        if self._task_branch != expected_branch:
            raise GitOperationError("A dedicated task branch must be active before committing")
        self._assert_current_task_branch()
        if not task_verification.passed:
            raise GitOperationError("Commit refused because task verification did not pass")
        if not message.strip():
            raise GitOperationError("Commit message must not be empty")

        before_evaluation = self._task_changes()
        changed_paths = sorted(before_evaluation)
        if not changed_paths:
            raise GitOperationError("Commit refused because the task change set is empty")
        self._assert_unstaged_task_changes(before_evaluation)

        self._run(["git", "diff", "--check", "--", *changed_paths])
        self_result = self_evaluator.verify_before_commit(changed_paths)
        if not self_result.passed:
            raise GitOperationError(
                "Commit refused because agent self-evaluation failed: "
                + "; ".join(self_result.failures)
            )
        after_evaluation = self._task_changes()
        if after_evaluation != before_evaluation:
            raise GitOperationError("Task change set changed while self-evaluation was running")

        self._run(["git", "add", "--", *changed_paths])
        staged_paths = self._read_staged_paths()
        if set(staged_paths) != set(changed_paths):
            raise GitOperationError(
                "Staged Git change set does not match the validated task change set"
            )
        result = self._run(["git", "commit", "-m", message])
        commit = result.stdout.strip()
        if not commit:
            raise GitOperationError("Git commit completed without returning commit output")
        return commit

    def _task_changes(self) -> dict[str, str]:
        self._assert_current_task_branch()
        current = self._read_status()
        for path, baseline_state in self._baseline_status.items():
            if current.get(path) != baseline_state:
                raise GitOperationError(
                    f"pre-existing unrelated change was modified or removed: {path}"
                )
        return {
            path: state
            for path, state in current.items()
            if path not in self._baseline_status
        }

    def _assert_unstaged_task_changes(self, status: dict[str, str]) -> None:
        for path, state in status.items():
            if state == "??":
                continue
            if state[0] != " " or any(flag in state for flag in ("R", "C", "U", "!")):
                raise GitOperationError(
                    f"Ambiguous staged, renamed, or conflicted change: {path}"
                )

    def _read_status(self) -> dict[str, str]:
        result = self._run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"]
        )
        statuses: dict[str, str] = {}
        for entry in result.stdout.split("\0"):
            if not entry:
                continue
            if len(entry) < 4 or entry[2] != " ":
                raise GitOperationError("Git returned an ambiguous porcelain status entry")
            state = entry[:2]
            path = entry[3:]
            if not path or any(flag in state for flag in ("R", "C", "U", "!")):
                raise GitOperationError("Git returned an ambiguous rename or conflict status")
            if path in statuses:
                raise GitOperationError(f"Git status contains a duplicate path: {path}")
            statuses[path] = state
        return statuses

    def _read_staged_paths(self) -> list[str]:
        result = self._run(["git", "diff", "--cached", "--name-only", "-z"])
        return [path for path in result.stdout.split("\0") if path]

    def _current_branch(self) -> str:
        return self._run(["git", "branch", "--show-current"]).stdout.strip()

    def _assert_current_task_branch(self) -> None:
        if self._task_branch is None:
            raise GitOperationError("Call ensure_task_branch before task Git operations")
        if self._current_branch() != self._task_branch:
            raise GitOperationError("The active branch is not the current task branch")

    def _run(self, argv: Sequence[str]):
        try:
            result = self.command_runner.run(argv, cwd=self.workspace)
        except CommandError as error:
            raise GitOperationError(f"Git command failed: {error}") from error
        if result.timed_out or result.exit_code != 0:
            raise GitOperationError(
                f"Git command failed: {argv!r}; exit={result.exit_code}; "
                f"timed_out={result.timed_out}; stderr={result.stderr!r}"
            )
        return result

    def _workspace_file(self, git_path: str) -> Path:
        path = self.workspace.joinpath(*PurePosixPath(git_path).parts)
        if path.is_symlink():
            raise GitOperationError(f"Refusing to read a symlink change: {git_path}")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.workspace)
        except (OSError, ValueError) as error:
            raise GitOperationError(f"Git path escapes the workspace: {git_path}") from error
        if not resolved.is_file():
            raise GitOperationError(f"Untracked Git path is not a regular file: {git_path}")
        return resolved

    @classmethod
    def _branch_for_task(cls, task_id: str) -> str:
        if not task_id or not task_id.strip():
            raise GitOperationError("Task id must not be empty")
        slug = cls._TASK_ID_PATTERN.sub("-", task_id.strip().lower()).strip("-")[:48].strip("-")
        if not slug:
            raise GitOperationError("Task id does not contain safe branch-name characters")
        return f"tuminh/task-{slug}"
