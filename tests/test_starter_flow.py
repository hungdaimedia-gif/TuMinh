from __future__ import annotations

import json
from pathlib import Path

from agent.config import Settings
from agent.models import AgentDecision, Subtask, Task, TaskPlan, TaskResult
from agent.orchestrator import Orchestrator
from agent.reflection import ReflectionStore
from agent.task_queue import TaskQueue
from main import AgentApplication
from tests.fakes import (
    FakePlanner,
    FakeToolRegistry,
    FakeVerifier,
    ScriptedModel,
)


class TestApp:
    __test__ = False

    def __init__(self, application: AgentApplication, github):
        self.application = application
        self.github = github

    def run_once(self) -> TaskResult:
        result = self.application.run_once()
        assert result is not None
        return result


class FakeGitHub:
    def __init__(self):
        self.calls = []

    def create_pull_request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "https://example.invalid/pull/1"


def sample_task() -> Task:
    return Task(
        id="offline-starter",
        title="Confirm the offline starter flow",
        description="Complete the configured offline starter task.",
        acceptance_criteria=["The fake model returns a final result."],
        source="local",
    )


def make_test_app(tmp_path: Path, model: ScriptedModel) -> TestApp:
    queue_file = tmp_path / "tasks" / "tasks.json"
    queue_file.parent.mkdir(parents=True)
    queue_file.write_text(
        json.dumps([sample_task().model_dump(mode="json")]),
        encoding="utf-8",
    )
    settings = Settings.from_env(
        {
            "WORKSPACE_ROOT": str(tmp_path),
            "TASKS_FILE": str(queue_file),
        }
    )
    queue = TaskQueue(settings.tasks_file, settings.task_state_file)
    task_plan = TaskPlan(
        task_id="offline-starter",
        subtasks=[
            Subtask(
                id="confirm",
                description="Return the offline result.",
                files=[],
                verification=["The fake verifier passes."],
            )
        ],
        verification_commands=[],
    )
    reflections = ReflectionStore(settings.reflection_file)
    orchestrator = Orchestrator(
        planner=FakePlanner(task_plan),
        model=model,
        tools=FakeToolRegistry(),
        settings=settings,
        verifier=FakeVerifier(),
        reflections=reflections,
    )
    github = FakeGitHub()
    application = AgentApplication(
        task_source=queue,
        task_queue=queue,
        orchestrator=orchestrator,
        settings=settings,
        git_operations=None,
        github_operations=github,
    )
    return TestApp(application, github)


def test_starter_flow_persists_success_without_external_services(tmp_path):
    app = make_test_app(
        tmp_path,
        ScriptedModel([AgentDecision(final_response="done")]),
    )

    result = app.run_once()

    assert result.status == "succeeded"
    assert json.loads(
        (tmp_path / "memory" / "task_state.json").read_text(encoding="utf-8")
    )["offline-starter"]["status"] == "succeeded"
    assert ReflectionStore(tmp_path / "memory" / "reflection.json").read_all()
    assert app.github.calls == []
