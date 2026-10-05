from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from agent.models import Task, TaskPlan, TaskResult, VerificationResult
from tools.command_runner import CommandResult


class ScriptedModel:
    def __init__(self, responses: Sequence[BaseModel | dict[str, Any]]):
        self.responses = list(responses)
        self.call_count = 0

    def complete_json(self, system_prompt, user_payload, response_type):
        if self.call_count >= len(self.responses):
            raise AssertionError("ScriptedModel received an unexpected extra request")
        response = self.responses[self.call_count]
        self.call_count += 1
        if isinstance(response, BaseModel):
            response = response.model_dump()
        return response_type.model_validate(response)


class FakePlanner:
    def __init__(self, plan: TaskPlan):
        self.plan = plan
        self.calls: list[Task] = []

    def create_plan(self, task: Task) -> TaskPlan:
        self.calls.append(task)
        return self.plan


class FakeVerifier:
    def __init__(self, result: VerificationResult | None = None):
        self.result = result or VerificationResult(passed=True, commands=[], failures=[])
        self.calls: list[tuple[TaskPlan, Sequence[str]]] = []

    def verify(self, plan: TaskPlan, changed_files: Sequence[str]) -> VerificationResult:
        self.calls.append((plan, changed_files))
        return self.result


class FakeReflectionStore:
    def __init__(self):
        self.reflections = []

    def append(self, reflection):
        self.reflections.append(reflection)


class FakeToolRegistry:
    def __init__(self, result: dict[str, object] | None = None):
        self.result = result or {"ok": True}
        self.calls = []

    def execute(self, call):
        self.calls.append(call)
        return self.result


class FakeCommandRunner:
    def __init__(self, exit_code: int = 0):
        self.exit_code = exit_code
        self.calls: list[list[str]] = []

    def run(self, argv, cwd):
        self.calls.append(list(argv))
        return CommandResult(
            argv=list(argv),
            exit_code=self.exit_code,
            stdout="",
            stderr="",
            timed_out=False,
        )
