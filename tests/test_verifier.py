from agent.models import Subtask, TaskPlan
from agent.verifier import CommandVerifier
from tests.fakes import FakeCommandRunner
from tools.command_runner import CommandResult


def plan(commands):
    return TaskPlan(
        task_id="task-1",
        subtasks=[
            Subtask(id="verify", description="Verify task.", files=[], verification=[])
        ],
        verification_commands=commands,
    )


def test_command_verifier_passes_when_all_commands_succeed(tmp_path):
    runner = FakeCommandRunner()
    verifier = CommandVerifier(runner)

    result = verifier.verify(plan([["python", "-m", "pytest", "-q"]]), [])

    assert result.passed is True
    assert result.commands == [["python", "-m", "pytest", "-q"]]
    assert runner.calls == [["python", "-m", "pytest", "-q"]]


def test_command_verifier_fails_when_no_commands_configured():
    result = CommandVerifier(FakeCommandRunner()).verify(plan([]), [])

    assert result.passed is False
    assert result.failures


def test_command_verifier_collects_nonzero_exit_status():
    result = CommandVerifier(FakeCommandRunner(exit_code=3)).verify(
        plan([["python", "-m", "pytest", "-q"]]),
        [],
    )

    assert result.passed is False
    assert "3" in result.failures[0]


def test_command_verifier_collects_timeout():
    runner = FakeCommandRunner(timed_out=True)

    result = CommandVerifier(runner).verify(plan([["python", "-m", "pytest"]]), [])

    assert result.passed is False
    assert "timed out" in result.failures[0].lower()


def test_command_verifier_surfaces_allowlist_rejection():
    class RejectingRunner:
        def run(self, argv, cwd):
            from tools.command_runner import CommandError

            raise CommandError("Command is not allowlisted")

    result = CommandVerifier(RejectingRunner()).verify(
        plan([["git", "status"]]),
        [],
    )

    assert result.passed is False
    assert "allowlisted" in result.failures[0]


def test_command_verifier_reports_every_failed_command():
    class FailingRunner:
        def run(self, argv, cwd):
            return CommandResult(
                argv=list(argv),
                exit_code=2,
                stdout="",
                stderr="bad",
                timed_out=False,
            )

    result = CommandVerifier(FailingRunner()).verify(
        plan([["pytest", "-q"], ["python", "-m", "evals.run"]]),
        [],
    )

    assert result.passed is False
    assert len(result.failures) == 2
