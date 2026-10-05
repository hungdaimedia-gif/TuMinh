import sys

import pytest

from tools.command_runner import CommandError, CommandRunner


def test_command_runner_executes_allowlisted_argv_without_shell(tmp_path):
    runner = CommandRunner([[sys.executable, "-c"]], timeout_seconds=3)

    result = runner.run(
        [sys.executable, "-c", "print('safe')"],
        cwd=tmp_path,
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "safe"
    assert result.stderr == ""
    assert result.timed_out is False


def test_command_runner_rejects_unlisted_executable(tmp_path):
    runner = CommandRunner([[sys.executable, "-c"]], timeout_seconds=2)

    with pytest.raises(CommandError, match="not allowlisted"):
        runner.run(["git", "status"], cwd=tmp_path)


def test_command_runner_rejects_unapproved_arguments(tmp_path):
    runner = CommandRunner(
        [[sys.executable, "-c", "print('approved')"]],
        timeout_seconds=2,
    )

    with pytest.raises(CommandError, match="not allowlisted"):
        runner.run([sys.executable, "-c", "print('different')"], cwd=tmp_path)


def test_command_runner_rejects_empty_argv(tmp_path):
    runner = CommandRunner([], timeout_seconds=2)

    with pytest.raises(CommandError, match="non-empty"):
        runner.run([], cwd=tmp_path)


def test_nonzero_command_surfaces_exit_status_and_output(tmp_path):
    runner = CommandRunner([[sys.executable, "-c"]], timeout_seconds=2)

    with pytest.raises(CommandError) as raised:
        runner.run(
            [sys.executable, "-c", "import sys; print('failed'); sys.exit(3)"],
            cwd=tmp_path,
        )

    assert raised.value.result is not None
    assert raised.value.result.exit_code == 3
    assert raised.value.result.stdout.strip() == "failed"
    assert raised.value.result.timed_out is False


def test_timeout_surfaces_timed_out_result(tmp_path):
    runner = CommandRunner([[sys.executable, "-c"]], timeout_seconds=1)

    with pytest.raises(CommandError) as raised:
        runner.run(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=tmp_path,
        )

    assert raised.value.result is not None
    assert raised.value.result.timed_out is True
    assert raised.value.result.exit_code is None
