from tests.fakes import FakeCommandRunner
from agent.self_eval import SelfEvaluator


def test_agent_change_requires_tests_and_evals(tmp_path):
    runner = FakeCommandRunner(exit_code=1)
    evaluator = SelfEvaluator(tmp_path, command_runner=runner)

    result = evaluator.verify_before_commit(["agent/orchestrator.py"])

    assert result.passed is False
    assert len(result.commands) == 2
    assert runner.calls == [
        ["python", "-m", "pytest", "-q"],
        ["python", "-m", "evals.run"],
    ]


def test_product_only_change_skips_agent_eval(tmp_path):
    runner = FakeCommandRunner(exit_code=0)
    evaluator = SelfEvaluator(tmp_path, command_runner=runner)

    result = evaluator.verify_before_commit(["src/product.py"])

    assert result.passed is True
    assert result.commands == []
    assert runner.calls == []


def test_every_agent_surface_requires_self_evaluation(tmp_path):
    for changed_file in (
        "main.py",
        "pyproject.toml",
        "tools/registry.py",
        "tests/test_models.py",
        "evals/cases.json",
        "skills/approved/safe.md",
    ):
        runner = FakeCommandRunner()
        result = SelfEvaluator(tmp_path, runner).verify_before_commit([changed_file])
        assert result.passed is True
        assert len(runner.calls) == 2


def test_self_evaluation_collects_each_failing_gate(tmp_path):
    runner = FakeCommandRunner(exit_code=4)

    result = SelfEvaluator(tmp_path, runner).verify_before_commit(["agent/config.py"])

    assert result.passed is False
    assert len(result.failures) == 2
