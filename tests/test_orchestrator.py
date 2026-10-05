from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from agent.config import Settings
from agent.model_client import TransientModelError
from agent.models import (
    AgentDecision,
    Subtask,
    Task,
    TaskPlan,
    ToolCall,
    VerificationResult,
)
from agent.orchestrator import OrchestrationError, Orchestrator
from tests.fakes import (
    FakePlanner,
    FakeReflectionStore,
    FakeSelfEvaluator,
    FakeToolRegistry,
    FakeVerifier,
    ScriptedModel,
)


def sample_task() -> Task:
    return Task(
        id="task-1",
        title="Add a safe check",
        description="Make a small change.",
        acceptance_criteria=["The focused test passes."],
        source="local",
    )


def sample_plan() -> TaskPlan:
    return TaskPlan(
        task_id="task-1",
        subtasks=[
            Subtask(
                id="change",
                description="Update the requested file.",
                files=["README.md"],
                verification=["Run focused tests."],
            )
        ],
        verification_commands=[["python", "-m", "pytest", "-q"]],
    )


def build_orchestrator(
    max_steps: int,
    decisions: Sequence[AgentDecision],
    *,
    verifier: FakeVerifier | None = None,
):
    import tempfile

    settings = Settings.from_env(
        {
            "WORKSPACE_ROOT": tempfile.gettempdir(),
            "MAX_STEPS": str(max_steps),
            "MAX_CORRECTIONS": "2",
        }
    )
    model = ScriptedModel(decisions)
    planner = FakePlanner(sample_plan())
    reflection_store = FakeReflectionStore()
    tools = FakeToolRegistry()
    verifier = verifier or FakeVerifier()
    orchestrator = Orchestrator(
        planner=planner,
        model=model,
        tools=tools,
        settings=settings,
        verifier=verifier,
        reflections=reflection_store,
    )
    return orchestrator, model, planner, tools, verifier, reflection_store


def test_orchestrator_runs_tool_then_verifies_success():
    decisions = [
        AgentDecision(
            tool_call=ToolCall(name="write_file", arguments={"path": "README.md"})
        ),
        AgentDecision(final_response="Updated README and verified it."),
    ]
    orchestrator, model, planner, tools, verifier, reflections = build_orchestrator(
        max_steps=3,
        decisions=decisions,
    )

    result = orchestrator.run(sample_task())

    assert result.status == "succeeded"
    assert result.changed_files == ["README.md"]
    assert planner.calls == [sample_task()]
    assert model.call_count == 2
    assert len(tools.calls) == 1
    assert verifier.calls[0][1] == ["README.md"]
    assert [reflection.outcome for reflection in reflections.reflections] == ["succeeded"]


def test_orchestrator_stops_at_max_steps():
    decisions = [
        AgentDecision(tool_call=ToolCall(name="read_file", arguments={"path": "README.md"}))
        for _ in range(3)
    ]
    orchestrator, model, _, _, verifier, reflections = build_orchestrator(2, decisions)

    result = orchestrator.run(sample_task())

    assert result.status == "failed"
    assert model.call_count == 2
    assert verifier.calls == []
    assert reflections.reflections[0].outcome == "failed"


def test_invalid_model_action_returns_failed_result():
    orchestrator, _, _, _, _, reflections = build_orchestrator(
        2,
        [{"tool_call": None, "final_response": None}],
    )

    result = orchestrator.run(sample_task())

    assert result.status == "failed"
    assert "Exactly one" in result.error
    assert reflections.reflections[0].outcome == "failed"


def test_failed_verification_allows_one_bounded_correction():
    decisions = [
        AgentDecision(final_response="Attempt one."),
        AgentDecision(
            tool_call=ToolCall(name="write_file", arguments={"path": "README.md"})
        ),
        AgentDecision(final_response="Corrected and verified."),
    ]
    verifier = FakeVerifier(
        [
            VerificationResult(passed=False, commands=[["pytest", "-q"]], failures=["failed"]),
            VerificationResult(passed=True, commands=[["pytest", "-q"]], failures=[]),
        ]
    )
    orchestrator, model, _, _, verifier, _ = build_orchestrator(
        5,
        decisions,
        verifier=verifier,
    )
    orchestrator.settings = Settings.from_env(
        {
            "WORKSPACE_ROOT": str(orchestrator.settings.workspace_root),
            "MAX_STEPS": "5",
            "MAX_CORRECTIONS": "1",
        }
    )

    result = orchestrator.run(sample_task())

    assert result.status == "succeeded"
    assert model.call_count == 3
    assert len(verifier.calls) == 2


def test_repeated_verification_failure_stops_at_correction_limit():
    decisions = [
        AgentDecision(final_response="Attempt one."),
        AgentDecision(final_response="Attempt two."),
        AgentDecision(final_response="Must not run."),
    ]
    verifier = FakeVerifier(
        [
            VerificationResult(passed=False, commands=[["pytest"]], failures=["failed"]),
            VerificationResult(passed=False, commands=[["pytest"]], failures=["still failing"]),
        ]
    )
    orchestrator, model, _, _, verifier, _ = build_orchestrator(
        5,
        decisions,
        verifier=verifier,
    )
    orchestrator.settings = Settings.from_env(
        {
            "WORKSPACE_ROOT": str(orchestrator.settings.workspace_root),
            "MAX_STEPS": "5",
            "MAX_CORRECTIONS": "1",
        }
    )

    result = orchestrator.run(sample_task())

    assert result.status == "failed"
    assert model.call_count == 2
    assert len(verifier.calls) == 2


def test_tool_exception_fails_task_and_records_reflection():
    class BrokenTools:
        def execute(self, call):
            raise RuntimeError("tool unavailable")

    orchestrator, _, _, _, _, reflections = build_orchestrator(
        3,
        [
            AgentDecision(
                tool_call=ToolCall(name="read_file", arguments={"path": "README.md"})
            )
        ],
    )
    orchestrator.tools = BrokenTools()

    result = orchestrator.run(sample_task())

    assert result.status == "failed"
    assert "tool unavailable" in result.error
    assert reflections.reflections[0].outcome == "failed"


def test_model_retries_only_transient_failures_with_limit():
    class FlakyModel:
        def __init__(self):
            self.call_count = 0

        def complete_json(self, system_prompt, user_payload, response_type):
            self.call_count += 1
            if self.call_count == 1:
                raise TransientModelError("temporary connection problem")
            return response_type.model_validate(
                {"final_response": "done", "tool_call": None}
            )

    orchestrator, _, _, _, _, reflections = build_orchestrator(
        1,
        [AgentDecision(final_response="placeholder")],
    )
    model = FlakyModel()
    orchestrator.model = model
    orchestrator.settings = Settings.from_env(
        {
            "WORKSPACE_ROOT": str(orchestrator.settings.workspace_root),
            "MAX_STEPS": "1",
            "MAX_RETRIES": "1",
        }
    )

    result = orchestrator.run(sample_task())

    assert result.status == "succeeded"
    assert model.call_count == 2
    assert reflections.reflections[0].outcome == "succeeded"


def test_model_retry_exhaustion_fails_task():
    class FailingModel:
        def __init__(self):
            self.call_count = 0

        def complete_json(self, system_prompt, user_payload, response_type):
            self.call_count += 1
            raise TransientModelError("connection unavailable")

    orchestrator, _, _, _, _, _ = build_orchestrator(
        2,
        [AgentDecision(final_response="unused")],
    )
    model = FailingModel()
    orchestrator.model = model
    orchestrator.settings = Settings.from_env(
        {
            "WORKSPACE_ROOT": str(orchestrator.settings.workspace_root),
            "MAX_RETRIES": "2",
        }
    )

    result = orchestrator.run(sample_task())

    assert result.status == "failed"
    assert model.call_count == 3


def test_reflection_failure_is_not_reported_as_success():
    class BrokenReflections:
        def append(self, reflection):
            raise OSError("reflection storage unavailable")

    orchestrator, _, _, _, _, _ = build_orchestrator(
        1,
        [AgentDecision(final_response="done")],
    )
    orchestrator.reflections = BrokenReflections()

    with pytest.raises(OrchestrationError, match="reflection"):
        orchestrator.run(sample_task())


def test_git_branch_is_created_before_planning_and_commit_follows_verification():
    events = []

    class FakeGitOperations:
        def ensure_task_branch(self, task_id):
            events.append(("branch", task_id))
            return "tuminh/task-task-1"

        def changed_files(self):
            events.append(("changed_files",))
            return ["README.md"]

        def commit_verified(self, task_id, message, *, task_verification, self_evaluator):
            events.append(("commit", task_id, task_verification.passed))
            assert self_evaluator is not None
            return "commit abc1234"

    orchestrator, model, planner, tools, verifier, reflections = build_orchestrator(
        2,
        [AgentDecision(final_response="Verified output.")],
    )
    orchestrator.git_operations = FakeGitOperations()
    orchestrator.self_evaluator = FakeSelfEvaluator(passed=True)

    result = orchestrator.run(sample_task())

    assert result.status == "succeeded"
    assert result.changed_files == ["README.md"]
    assert events[0] == ("branch", "task-1")
    assert events[-1] == ("commit", "task-1", True)
    assert planner.calls == [sample_task()]
    assert model.call_count == 1
    assert tools.calls == []
    assert verifier.calls[0][1] == ["README.md"]
    assert reflections.reflections[0].outcome == "succeeded"


def test_failed_task_verification_does_not_attempt_git_commit():
    class FakeGitOperations:
        commits = 0

        def ensure_task_branch(self, task_id):
            return "tuminh/task-task-1"

        def changed_files(self):
            return ["README.md"]

        def commit_verified(self, *args, **kwargs):
            self.commits += 1
            raise AssertionError("Commit must not run after failed verification")

    git = FakeGitOperations()
    verifier = FakeVerifier(
        VerificationResult(passed=False, commands=[["pytest"]], failures=["failed"])
    )
    orchestrator, _, _, _, _, _ = build_orchestrator(
        2,
        [AgentDecision(final_response="Not verified.")],
        verifier=verifier,
    )
    orchestrator.settings = Settings.from_env(
        {
            "WORKSPACE_ROOT": str(orchestrator.settings.workspace_root),
            "MAX_STEPS": "2",
            "MAX_CORRECTIONS": "1",
        }
    )
    orchestrator.git_operations = git
    orchestrator.self_evaluator = FakeSelfEvaluator(passed=True)

    result = orchestrator.run(sample_task())

    assert result.status == "failed"
    assert git.commits == 0
