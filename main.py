from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent.config import Settings, SettingsError
from agent.github_source import GitHubIssueSource, GitHubSourceError
from agent.model_client import ModelConfigurationError, OpenAICompatibleModelClient
from agent.models import Task, TaskResult
from agent.orchestrator import OrchestrationError, Orchestrator
from agent.planner import Planner
from agent.reflection import HumanApproval, ReflectionError, ReflectionStore
from agent.self_eval import SelfEvaluator
from agent.task_queue import TaskQueue, TaskQueueError
from agent.trajectory import TrajectoryJournal, TrajectoryError
from agent.verifier import CommandVerifier
from evals.run import evaluate_proposal
from tools.command_runner import CommandError, CommandResult, CommandRunner
from tools.github_ops import GitHubOperationError, GitHubOperations
from tools.git_ops import GitOperationError, GitOperations
from tools.registry import ToolRegistry
from tools.workspace import Workspace, WorkspaceError


class TaskSource(Protocol):
    def next_task(self) -> Task | None: ...


class ReadFileArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)


class WriteFileArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)
    content: str


class ListFilesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = "."


class RunCommandArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    argv: list[str] = Field(min_length=1)


class PendingPullRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    branch: str = Field(min_length=1)
    title: str = Field(min_length=1)
    body: str
    result: TaskResult

    @model_validator(mode="after")
    def validate_result(self) -> PendingPullRequest:
        if self.result.status != "succeeded" or self.result.task_id != self.task_id:
            raise ValueError("Pending pull request must reference a succeeded matching task")
        return self


class AgentApplication:
    def __init__(
        self,
        *,
        task_source: TaskSource,
        task_queue: TaskQueue,
        orchestrator: Orchestrator,
        settings: Settings,
        git_operations: GitOperations | None,
        github_operations: GitHubOperations | None = None,
    ):
        self.task_source = task_source
        self.task_queue = task_queue
        self.orchestrator = orchestrator
        self.settings = settings
        self.git_operations = git_operations
        self.github_operations = github_operations

    def run_once(self) -> TaskResult | None:
        if self.settings.create_pull_request:
            pending = self._read_pending_pull_request()
            if pending is not None:
                return self._publish_pending_pull_request(pending)
        task = self.task_source.next_task()
        if task is None:
            return None
        result = self.orchestrator.run(task)
        if result.status == "succeeded" and self.settings.create_pull_request:
            if self.github_operations is None or self.git_operations is None:
                raise GitHubOperationError(
                    "CREATE_PULL_REQUEST is enabled but Git operations are unavailable"
                )
            branch = self.git_operations.ensure_task_branch(task.id)
            pending = PendingPullRequest(
                task_id=task.id,
                branch=branch,
                title=task.title,
                body=result.summary,
                result=result,
            )
            self._write_pending_pull_request(pending)
            self.task_queue.save_result(task.id, result)
            return self._publish_pending_pull_request(pending)
        self.task_queue.save_result(task.id, result)
        return result

    def _read_pending_pull_request(self) -> PendingPullRequest | None:
        path = self.settings.pull_request_state_file
        self._assert_pr_state_path_contained()
        if not path.exists():
            return None
        try:
            return PendingPullRequest.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValidationError, ValueError) as error:
            raise GitHubOperationError(
                f"Could not read pending pull request state: {error}"
            ) from error

    def _write_pending_pull_request(self, pending: PendingPullRequest) -> None:
        path = self.settings.pull_request_state_file
        temporary_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._assert_pr_state_path_contained()
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(pending.model_dump_json())
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
        except OSError as error:
            raise GitHubOperationError(
                f"Could not persist pending pull request state: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

    def _publish_pending_pull_request(
        self,
        pending: PendingPullRequest,
    ) -> TaskResult:
        if self.github_operations is None:
            raise GitHubOperationError(
                "CREATE_PULL_REQUEST is enabled but GitHub operations are unavailable"
            )
        pull_request_url = self.github_operations.create_pull_request(
            branch=pending.branch,
            title=pending.title,
            body=pending.body,
        )
        result = pending.result.model_copy(
            update={"summary": f"{pending.result.summary}\nPull request: {pull_request_url}"}
        )
        self.task_queue.save_result(pending.task_id, result)
        try:
            self._assert_pr_state_path_contained()
            self.settings.pull_request_state_file.unlink(missing_ok=True)
        except OSError as error:
            raise GitHubOperationError(
                f"Pull request was created but pending state could not be cleared: {error}"
            ) from error
        return result

    def _assert_pr_state_path_contained(self) -> None:
        root = self.settings.workspace_root.resolve()
        path = self.settings.pull_request_state_file.resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise GitHubOperationError(
                "Pending pull request state must remain inside WORKSPACE_ROOT"
            ) from error


def create_task_source(
    settings: Settings,
    command_runner: CommandRunner,
) -> TaskQueue | GitHubIssueSource:
    if settings.task_source == "local":
        return TaskQueue(settings.tasks_file, settings.task_state_file)
    return GitHubIssueSource(
        command_runner,
        settings.task_state_file,
        repository=settings.github_repository,
        cwd=settings.workspace_root,
    )


def create_application(settings: Settings) -> AgentApplication:
    command_runner = CommandRunner(
        settings.allowed_argv_prefixes,
        timeout_seconds=settings.command_timeout_seconds,
    )
    agent_prefixes = _agent_command_prefixes(settings.allowed_argv_prefixes)
    agent_command_runner = CommandRunner(
        agent_prefixes,
        timeout_seconds=settings.command_timeout_seconds,
    )
    verifier_command_runner = CommandRunner(
        _verification_command_prefixes(settings.allowed_argv_prefixes),
        timeout_seconds=settings.command_timeout_seconds,
    )
    workspace = Workspace(settings.workspace_root)
    registry = ToolRegistry()

    def write_workspace_file(arguments: WriteFileArguments) -> dict[str, object]:
        workspace.write_file(arguments.path, arguments.content)
        return {"changed_files": [arguments.path]}

    registry.register(
        "read_file",
        ReadFileArguments,
        lambda arguments: {"content": workspace.read_file(arguments.path)},
    )
    registry.register(
        "write_file",
        WriteFileArguments,
        write_workspace_file,
    )
    registry.register(
        "list_files",
        ListFilesArguments,
        lambda arguments: {"files": workspace.list_files(arguments.path)},
    )
    registry.register(
        "run_command",
        RunCommandArguments,
        lambda arguments: _command_result(
            agent_command_runner.run(arguments.argv, cwd=settings.workspace_root)
        ),
    )

    secret_values = [settings.model_api_key] if settings.model_api_key else []
    model = OpenAICompatibleModelClient(settings)
    task_queue = TaskQueue(settings.tasks_file, settings.task_state_file)
    task_source = create_task_source(settings, command_runner)
    git_operations = GitOperations(settings.workspace_root, command_runner)
    self_evaluator = SelfEvaluator(settings.workspace_root, command_runner)
    reflections = ReflectionStore(settings.reflection_file, secret_values=secret_values)
    trajectory = TrajectoryJournal(
        settings.workspace_root / "memory" / "trajectories",
        secret_values=secret_values,
    )
    orchestrator = Orchestrator(
        planner=Planner(model),
        model=model,
        tools=registry,
        settings=settings,
        verifier=CommandVerifier(verifier_command_runner, cwd=settings.workspace_root),
        reflections=reflections,
        git_operations=git_operations,
        self_evaluator=self_evaluator,
        trajectory=trajectory,
    )
    github_operations = (
        GitHubOperations(
            command_runner,
            repository=settings.github_repository,
            cwd=settings.workspace_root,
        )
        if settings.create_pull_request
        else None
    )
    return AgentApplication(
        task_source=task_source,
        task_queue=task_queue,
        orchestrator=orchestrator,
        settings=settings,
        git_operations=git_operations,
        github_operations=github_operations,
    )


def request_skill_approval(
    input_fn: Callable[[str], str] = input,
) -> HumanApproval | None:
    response = input_fn("Approve and activate this skill proposal? [y/N] ")
    if response.strip().lower() not in {"y", "yes"}:
        return None
    return HumanApproval(
        approved_by=getpass.getuser(),
        confirmed_at=datetime.now(UTC),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    app_factory: Callable[[Settings], AgentApplication] | None = None,
    input_fn: Callable[[str], str] = input,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    if env is None:
        load_dotenv()
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        settings = Settings.from_env(env)
        if arguments.command == "approve-skill-proposal":
            return _approve_skill_proposal(
                Path(arguments.path),
                settings,
                input_fn=input_fn,
            )

        application = (app_factory or create_application)(settings)
        if arguments.poll:
            return _poll(application, settings, sleeper=sleeper)
        result = application.run_once()
        if result is None:
            print("No pending task.")
            return 0
        _print_result(result)
        return 0 if result.status == "succeeded" else 1
    except (
        CommandError,
        GitHubOperationError,
        GitHubSourceError,
        GitOperationError,
        ModelConfigurationError,
        OrchestrationError,
        ReflectionError,
        SettingsError,
        TaskQueueError,
        TrajectoryError,
        WorkspaceError,
        ValueError,
    ) as error:
        print(f"TuMinh error: {error}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TuMinh autonomous coding agent.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--once",
        action="store_true",
        help="Run one task (the default).",
    )
    mode.add_argument(
        "--poll",
        action="store_true",
        help="Poll the configured task source for a bounded number of cycles.",
    )
    subcommands = parser.add_subparsers(dest="command")
    approve = subcommands.add_parser(
        "approve-skill-proposal",
        help="Evaluate and explicitly approve an inert skill proposal.",
    )
    approve.add_argument("path", help="Path under skills/proposals/")
    return parser


def _approve_skill_proposal(
    path: Path,
    settings: Settings,
    *,
    input_fn: Callable[[str], str],
) -> int:
    evaluation = evaluate_proposal(path)
    if not evaluation.passed:
        print("Skill proposal evaluation failed:")
        for failure in evaluation.failures:
            print(f"- {failure}")
        return 1

    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReflectionError(f"Could not read skill proposal: {error}") from error
    print("Skill proposal:")
    print(content)
    print("Evaluation: passed")
    approval = request_skill_approval(input_fn=input_fn)
    if approval is None:
        print("Skill proposal was not activated.")
        return 1

    secret_values = [settings.model_api_key] if settings.model_api_key else []
    active_path = ReflectionStore(
        settings.reflection_file,
        secret_values=secret_values,
    ).activate_proposal(
        path,
        eval_result=evaluation,
        approval=approval,
    )
    print(f"Activated skill proposal: {active_path}")
    return 0


def _poll(
    application: AgentApplication,
    settings: Settings,
    *,
    sleeper: Callable[[float], None],
) -> int:
    failed = False
    for cycle in range(settings.max_poll_cycles):
        result = application.run_once()
        if result is not None:
            _print_result(result)
            failed = failed or result.status == "failed"
        if cycle + 1 < settings.max_poll_cycles:
            sleeper(settings.poll_interval_seconds)
    return 1 if failed else 0


def _print_result(result: TaskResult) -> None:
    print(f"Task {result.task_id}: {result.status}")
    print(result.summary)
    if result.error:
        print(f"Error: {result.error}", file=sys.stderr)


def _command_result(result: CommandResult) -> dict[str, object]:
    return {
        "argv": result.argv,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": result.timed_out,
    }


def _agent_command_prefixes(
    prefixes: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(prefix)
        for prefix in prefixes
        if prefix and prefix[0] not in {"git", "gh"}
    )


def _verification_command_prefixes(
    prefixes: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    git_mutations = {"add", "commit", "push", "switch", "checkout", "reset", "clean"}
    return tuple(
        tuple(prefix)
        for prefix in prefixes
        if prefix
        and prefix[0] != "gh"
        and not (prefix[0] == "git" and len(prefix) > 1 and prefix[1] in git_mutations)
    )


if __name__ == "__main__":
    raise SystemExit(main())
