import pytest
from pydantic import ValidationError

from agent.models import (
    AgentDecision,
    Task,
    TaskPlan,
    TaskResult,
    ToolCall,
    VerificationResult,
)


def test_task_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(
            {
                "id": "local-1",
                "title": "A safe task",
                "description": "Update one test.",
                "source": "unknown",
                "state": "pending",
            }
        )


def test_agent_decision_requires_exactly_one_outcome() -> None:
    call = ToolCall(name="read_file", arguments={"path": "README.md"})

    with pytest.raises(ValidationError):
        AgentDecision(tool_call=call, final_response="done")

    with pytest.raises(ValidationError):
        AgentDecision()


def test_agent_decision_accepts_a_valid_tool_call() -> None:
    call = ToolCall(name="read_file", arguments={"path": "README.md"})

    decision = AgentDecision(tool_call=call)

    assert decision.tool_call == call
    assert decision.final_response is None


def test_task_plan_requires_at_least_one_subtask() -> None:
    with pytest.raises(ValidationError):
        TaskPlan(task_id="task-1", subtasks=[], verification_commands=[])


def test_failed_task_result_requires_an_error_reason() -> None:
    with pytest.raises(ValidationError):
        TaskResult(
            task_id="task-1",
            status="failed",
            summary="No details",
            changed_files=[],
        )


def test_verification_result_failure_summary_passing() -> None:
    result = VerificationResult(passed=True, commands=[["pytest"]])
    assert result.failure_summary == "All verification commands passed."


def test_verification_result_failure_summary_failing_multiple() -> None:
    result = VerificationResult(
        passed=False,
        commands=[["pytest"]],
        failures=["first failure", "second failure"],
    )
    assert result.failure_summary == "2 verification failure(s): first failure; second failure"

