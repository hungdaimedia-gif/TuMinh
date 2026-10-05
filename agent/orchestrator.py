from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

from agent.config import Settings
from agent.model_client import ModelClient, TransientModelError
from agent.models import (
    AgentDecision,
    Reflection,
    Task,
    TaskPlan,
    TaskResult,
    VerificationResult,
)
from agent.planner import Planner
from agent.prompts import EXECUTION_SYSTEM_PROMPT
from agent.trajectory import TrajectoryError
from agent.verifier import Verifier
from tools.git_ops import GitOperations, SelfEvaluator as GitSelfEvaluator
from tools.registry import ToolRegistry


class ReflectionStore(Protocol):
    def append(self, reflection: Reflection) -> None: ...


class TrajectoryRecorder(Protocol):
    def append(self, task_id: str, event_type: str, data: dict[str, object]) -> None: ...


class OrchestrationError(RuntimeError):
    """Raised when the task outcome cannot be recorded reliably."""


TResult = TypeVar("TResult")


class Orchestrator:
    def __init__(
        self,
        planner: Planner,
        model: ModelClient,
        tools: ToolRegistry,
        settings: Settings,
        verifier: Verifier,
        reflections: ReflectionStore,
        git_operations: GitOperations | None = None,
        self_evaluator: GitSelfEvaluator | None = None,
        trajectory: TrajectoryRecorder | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        if (git_operations is None) != (self_evaluator is None):
            raise ValueError("Git operations and a self-evaluator must be configured together")
        self.planner = planner
        self.model = model
        self.tools = tools
        self.settings = settings
        self.verifier = verifier
        self.reflections = reflections
        self.git_operations = git_operations
        self.self_evaluator = self_evaluator
        self.trajectory = trajectory
        self._sleep_fn = sleep_fn

    def run(self, task: Task) -> TaskResult:
        changed_files: list[str] = []
        try:
            return self._run(task, changed_files)
        except OrchestrationError:
            raise
        except Exception as error:
            return self._failed(
                task,
                reason=str(error) or error.__class__.__name__,
                changed_files=changed_files,
                failure_category="execution_error",
                record_trajectory=not isinstance(error, TrajectoryError),
            )

    def _run(self, task: Task, changed_files: list[str]) -> TaskResult:
        self._append_trajectory(task.id, "task", task.model_dump(mode="json"))
        if self.git_operations is not None:
            self.git_operations.ensure_task_branch(task.id)
        plan = self._with_model_retries(lambda: self.planner.create_plan(task))
        if plan.task_id != task.id:
            raise ValueError("Planner returned a plan for a different task")
        self._append_trajectory(task.id, "plan", plan.model_dump(mode="json"))

        history: list[dict[str, object]] = [
            {"type": "plan", "plan": plan.model_dump(mode="json")}
        ]
        available_tools = self.tools.describe_tools()
        corrections = 0
        for _ in range(self.settings.max_steps):
            payload = json.dumps(
                {
                    "task": task.model_dump(mode="json"),
                    "plan": plan.model_dump(mode="json"),
                    "history": history,
                    "available_tools": available_tools,
                },
                ensure_ascii=False,
            )
            decision = self._with_model_retries(
                lambda: self.model.complete_json(
                    EXECUTION_SYSTEM_PROMPT,
                    payload,
                    AgentDecision,
                )
            )
            history.append({"type": "decision", "decision": decision.model_dump(mode="json")})
            self._append_trajectory(
                task.id,
                "decision",
                {"decision": decision.model_dump(mode="json")},
            )

            if decision.tool_call is not None:
                tool_call = decision.tool_call
                self._append_trajectory(
                    task.id,
                    "tool_call",
                    {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                )
                try:
                    tool_result = self.tools.execute(tool_call)
                except Exception as error:
                    self._append_trajectory(
                        task.id,
                        "tool_result",
                        {"error": str(error) or error.__class__.__name__},
                    )
                    raise
                self._collect_changed_files(
                    changed_files,
                    tool_call.name,
                    tool_call.arguments,
                    tool_result,
                )
                self._append_trajectory(
                    task.id,
                    "tool_result",
                    {"result": tool_result},
                )
                history.append({"type": "tool_result", "result": tool_result})
                continue

            if self.git_operations is not None:
                changed_files[:] = self.git_operations.changed_files()
                if self.trajectory is not None:
                    self._append_trajectory(
                        task.id,
                        "patch",
                        {"diff": self.git_operations.diff_text()},
                    )
            verification = self.verifier.verify(plan, changed_files)
            self._append_trajectory(
                task.id,
                "verification",
                {"result": verification.model_dump(mode="json")},
            )
            history.append(
                {
                    "type": "verification",
                    "result": verification.model_dump(mode="json"),
                }
            )
            if verification.passed:
                if self.git_operations is not None and self.self_evaluator is not None:
                    self.git_operations.commit_verified(
                        task.id,
                        self._commit_message(task.title),
                        task_verification=verification,
                        self_evaluator=self.self_evaluator,
                    )
                return self._succeeded(
                    task,
                    summary=decision.final_response or "",
                    changed_files=changed_files,
                )
            if corrections >= self.settings.max_corrections:
                return self._failed(
                    task,
                    reason="Verification failed: " + "; ".join(verification.failures),
                    changed_files=changed_files,
                    failure_category="verification_failed",
                )
            corrections += 1
            history.append(
                {
                    "type": "correction_required",
                    "failures": verification.failures,
                    "correction": corrections,
                }
            )

        return self._failed(
            task,
            reason=f"Execution exceeded the {self.settings.max_steps}-step limit.",
            changed_files=changed_files,
            failure_category="max_steps",
        )

    def _with_model_retries(self, action: Callable[[], TResult]) -> TResult:
        retries = 0
        while True:
            try:
                return action()
            except TransientModelError:
                if retries >= self.settings.max_retries:
                    raise
                retries += 1
                self._sleep_fn(10 * retries)

    def _succeeded(
        self,
        task: Task,
        *,
        summary: str,
        changed_files: Sequence[str],
    ) -> TaskResult:
        self._append_reflection(
            Reflection(
                task_id=task.id,
                outcome="succeeded",
                lessons=[summary],
            )
        )
        return TaskResult(
            task_id=task.id,
            status="succeeded",
            summary=summary,
            changed_files=list(changed_files),
            error=None,
        )

    def _failed(
        self,
        task: Task,
        *,
        reason: str,
        changed_files: Sequence[str],
        failure_category: str,
        record_trajectory: bool = True,
    ) -> TaskResult:
        self._append_reflection(
            Reflection(
                task_id=task.id,
                outcome="failed",
                lessons=[reason],
                failure_category=failure_category,
            ),
            record_trajectory=record_trajectory,
        )
        return TaskResult(
            task_id=task.id,
            status="failed",
            summary="Task failed.",
            changed_files=list(changed_files),
            error=reason,
        )

    def _append_reflection(
        self,
        reflection: Reflection,
        *,
        record_trajectory: bool = True,
    ) -> None:
        try:
            self.reflections.append(reflection)
        except Exception as error:
            raise OrchestrationError(
                f"Could not persist task reflection: {error}"
            ) from error
        if self.trajectory is not None and record_trajectory:
            self._append_trajectory(
                reflection.task_id,
                "reflection",
                {
                    "outcome": reflection.outcome,
                    "lessons": reflection.lessons,
                    "failure_category": reflection.failure_category,
                },
            )

    def _append_trajectory(
        self,
        task_id: str,
        event_type: str,
        data: dict[str, object],
    ) -> None:
        if self.trajectory is not None:
            self.trajectory.append(task_id, event_type, data)

    @staticmethod
    def _collect_changed_files(
        changed_files: list[str],
        tool_name: str,
        arguments: dict[str, object],
        result: dict[str, object],
    ) -> None:
        candidates: list[object] = []
        if tool_name == "write_file":
            candidates.append(arguments.get("path"))
        tool_changed_files = result.get("changed_files")
        if isinstance(tool_changed_files, list):
            candidates.extend(tool_changed_files)
        for candidate in candidates:
            if isinstance(candidate, str) and candidate not in changed_files:
                changed_files.append(candidate)

    @staticmethod
    def _commit_message(title: str) -> str:
        concise_title = " ".join(title.split())[:64]
        return f"feat: {concise_title}"
