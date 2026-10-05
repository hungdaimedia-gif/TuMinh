from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from distill.benchmark import BenchmarkError, BenchmarkRunner, is_candidate_eligible
from distill.schema import EvalCase, ModelScore
from tests.distill_fakes import complete_benchmark_metadata, synthetic_eval_cases


class FakeEvaluator:
    def __init__(
        self,
        scores: dict[str, tuple[dict[str, bool], dict[str, float]]],
        *,
        error: Exception | None = None,
    ):
        self.scores = scores
        self.error = error
        self.calls: list[
            tuple[str, tuple[EvalCase, ...], Mapping[str, object], Mapping[str, object]]
        ] = []

    def evaluate(
        self,
        model_name: str,
        cases: tuple[EvalCase, ...],
        *,
        tool_limits: Mapping[str, object] | None = None,
        decoding_config: Mapping[str, object] | None = None,
    ) -> ModelScore:
        self.calls.append((model_name, cases, tool_limits, decoding_config))
        if self.error is not None:
            raise self.error
        results, resources = self.scores[model_name]
        return ModelScore(
            model_id=model_name,
            model_revision=f"{model_name}-revision",
            case_results=results,
            passed_count=sum(results.values()),
            total_count=len(results),
            pass_rate=sum(results.values()) / len(results),
            resource_measurements=resources,
        )


class MutatingEvaluator(FakeEvaluator):
    def evaluate(
        self,
        model_name: str,
        cases: tuple[EvalCase, ...],
        *,
        tool_limits: Mapping[str, object],
        decoding_config: Mapping[str, object],
    ) -> ModelScore:
        tool_limits["max_tool_calls"] = 100
        return super().evaluate(
            model_name,
            cases,
            tool_limits=tool_limits,
            decoding_config=decoding_config,
        )


def score_results(
    cases: tuple[EvalCase, ...],
    *,
    pass_count: int,
    failed_safety_case: str | None = None,
) -> dict[str, bool]:
    results = {case.case_id: True for case in cases}
    if failed_safety_case is not None:
        results[failed_safety_case] = False
    non_safety_cases = [case.case_id for case in cases if not case.safety_critical]
    failures_needed = len(cases) - pass_count - (failed_safety_case is not None)
    for case_id in non_safety_cases[:failures_needed]:
        results[case_id] = False
    return results


def evaluator_for(
    cases: tuple[EvalCase, ...],
    baseline_pass_count: int,
    candidate_pass_count: int,
    *,
    failed_safety_case: str | None = None,
) -> FakeEvaluator:
    return FakeEvaluator(
        {
            "baseline-model": (
                score_results(cases, pass_count=baseline_pass_count),
                {"duration_seconds": 2.0, "peak_memory_mb": 100.0},
            ),
            "candidate-model": (
                score_results(
                    cases,
                    pass_count=candidate_pass_count,
                    failed_safety_case=failed_safety_case,
                ),
                {"duration_seconds": 1.5, "peak_memory_mb": 90.0},
            ),
        }
    )


def test_equal_pass_rates_are_eligible_and_use_the_same_pinned_cases():
    cases = tuple(synthetic_eval_cases())
    evaluator = evaluator_for(cases, 6, 6)
    metadata = complete_benchmark_metadata()
    metadata.tool_limits["nested"] = {"allowed": [1, 2]}

    report = BenchmarkRunner(evaluator, cases).compare(
        "baseline-model", "candidate-model", metadata
    )

    assert report.eligible_for_promotion is True
    assert report.baseline_score.pass_rate == report.candidate_score.pass_rate
    assert evaluator.calls[0][1] is evaluator.calls[1][1]
    assert [call[0] for call in evaluator.calls] == ["baseline-model", "candidate-model"]
    expected_tool_limits = {
        **report.metadata.tool_limits,
        "nested": {"allowed": (1, 2)},
    }
    assert evaluator.calls[0][2] == expected_tool_limits
    assert evaluator.calls[1][2] == expected_tool_limits
    assert evaluator.calls[0][3] == report.metadata.decoding_config
    assert evaluator.calls[1][3] == report.metadata.decoding_config
    assert evaluator.calls[0][2] is evaluator.calls[1][2]
    assert evaluator.calls[0][3] is evaluator.calls[1][3]
    assert evaluator.calls[0][2]["nested"] is evaluator.calls[1][2]["nested"]
    with pytest.raises(TypeError, match="mappingproxy"):
        evaluator.calls[0][2]["max_tool_calls"] = 1
    with pytest.raises(TypeError, match="mappingproxy"):
        evaluator.calls[1][3]["temperature"] = 1
    with pytest.raises(TypeError, match="mappingproxy"):
        evaluator.calls[0][2]["nested"]["allowed"] = [9]
    with pytest.raises(TypeError):
        evaluator.calls[1][2]["nested"]["allowed"][0] = 9
    assert report.eval_case_ids == [case.case_id for case in cases]
    assert report.safety_results == {
        case.case_id: True for case in cases if case.safety_critical
    }
    assert report.eval_set_hash
    assert report.metadata.tool_limits
    assert report.metadata.decoding_config
    assert report.metadata.software_metadata
    assert report.metadata.hardware_metadata


def test_evaluator_cannot_change_shared_settings_to_create_a_mismatch():
    cases = tuple(synthetic_eval_cases())
    evaluator = MutatingEvaluator(
        {
            "baseline-model": (
                score_results(cases, pass_count=6),
                {"duration_seconds": 2.0, "peak_memory_mb": 100.0},
            ),
            "candidate-model": (
                score_results(cases, pass_count=6),
                {"duration_seconds": 1.5, "peak_memory_mb": 90.0},
            ),
        }
    )

    with pytest.raises(BenchmarkError, match="evaluation failed") as error:
        BenchmarkRunner(evaluator, cases).compare(
            "baseline-model", "candidate-model", complete_benchmark_metadata()
        )

    assert isinstance(error.value.__cause__, TypeError)
    assert "mappingproxy" in str(error.value.__cause__)


def test_lower_candidate_pass_rate_is_ineligible():
    cases = tuple(synthetic_eval_cases())
    report = BenchmarkRunner(evaluator_for(cases, 6, 5), cases).compare(
        "baseline-model", "candidate-model", complete_benchmark_metadata()
    )

    assert report.eligible_for_promotion is False


def test_failed_safety_critical_case_blocks_candidate_despite_higher_score():
    cases = tuple(synthetic_eval_cases())
    safety_case = next(case.case_id for case in cases if case.safety_critical)
    report = BenchmarkRunner(
        evaluator_for(cases, 4, 7, failed_safety_case=safety_case), cases
    ).compare("baseline-model", "candidate-model", complete_benchmark_metadata())

    assert report.candidate_score.pass_rate > report.baseline_score.pass_rate
    assert report.safety_results[safety_case] is False
    assert report.eligible_for_promotion is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_model_id", None),
        ("candidate_model_revision", ""),
        ("dataset_hash", None),
        ("eval_set_version", ""),
    ],
)
def test_missing_model_dataset_or_eval_metadata_is_ineligible(field, value):
    cases = tuple(synthetic_eval_cases())
    metadata = complete_benchmark_metadata().model_copy(update={field: value})
    report = BenchmarkRunner(evaluator_for(cases, 6, 6), cases).compare(
        "baseline-model", "candidate-model", metadata
    )

    assert report.eligible_for_promotion is False


def test_missing_safety_results_are_ineligible():
    metadata = complete_benchmark_metadata()

    assert (
        is_candidate_eligible(
            0.75,
            0.75,
            {},
            metadata,
        )
        is False
    )


def test_malformed_dataset_hash_is_ineligible():
    cases = tuple(synthetic_eval_cases())
    report = BenchmarkRunner(evaluator_for(cases, 6, 6), cases).compare(
        "baseline-model", "candidate-model", complete_benchmark_metadata()
    )
    metadata = report.metadata.model_copy(update={"dataset_hash": "not-a-sha256"})
    malformed_report = report.model_copy(update={"metadata": metadata})

    assert malformed_report.eligible_for_promotion is False


def test_missing_candidate_case_results_are_ineligible():
    cases = tuple(synthetic_eval_cases())
    baseline_results = score_results(cases, pass_count=6)
    candidate_results = score_results(cases[:-1], pass_count=5)
    evaluator = FakeEvaluator(
        {
            "baseline-model": (
                baseline_results,
                {"duration_seconds": 2.0, "peak_memory_mb": 100.0},
            ),
            "candidate-model": (
                candidate_results,
                {"duration_seconds": 1.5, "peak_memory_mb": 90.0},
            ),
        }
    )
    report = BenchmarkRunner(evaluator, cases).compare(
        "baseline-model", "candidate-model", complete_benchmark_metadata()
    )

    assert report.eligible_for_promotion is False


def test_safety_results_for_unknown_case_are_ineligible():
    cases = tuple(synthetic_eval_cases())
    report = BenchmarkRunner(evaluator_for(cases, 6, 6), cases).compare(
        "baseline-model", "candidate-model", complete_benchmark_metadata()
    )
    unsafe_report = report.model_copy(
        update={
            "metadata": report.metadata.model_copy(
                update={"safety_critical_case_ids": ["unknown-case"]}
            ),
            "safety_case_ids": ["unknown-case"],
            "safety_results": {"unknown-case": True},
        }
    )

    assert unsafe_report.eligible_for_promotion is False


def test_model_score_rejects_inconsistent_counts_or_missing_resources():
    with pytest.raises(ValidationError, match="passed_count"):
        ModelScore(
            model_id="candidate-model",
            model_revision="candidate-model-revision",
            case_results={"case-1": True},
            passed_count=0,
            total_count=1,
            pass_rate=0,
            resource_measurements={"duration_seconds": 1},
        )

    with pytest.raises(ValidationError):
        ModelScore(
            model_id="candidate-model",
            model_revision="candidate-model-revision",
            case_results={"case-1": True},
            passed_count=1,
            total_count=1,
            pass_rate=1,
            resource_measurements={},
        )


def test_model_score_rejects_non_boolean_case_results():
    with pytest.raises(ValidationError):
        ModelScore(
            model_id="candidate-model",
            model_revision="candidate-model-revision",
            case_results={"case-1": "yes"},
            passed_count=1,
            total_count=1,
            pass_rate=1,
            resource_measurements={"duration_seconds": 1},
        )


def test_evaluator_error_fails_closed():
    cases = tuple(synthetic_eval_cases())
    evaluator = FakeEvaluator({}, error=RuntimeError("synthetic evaluator failure"))

    with pytest.raises(BenchmarkError, match="evaluation failed"):
        BenchmarkRunner(evaluator, cases).compare(
            "baseline-model", "candidate-model", complete_benchmark_metadata()
        )


def test_eval_cases_fixture_is_valid_and_has_safety_critical_cases():
    fixture = Path(__file__).parents[1] / "distill" / "evals" / "cases.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))

    cases = [EvalCase.model_validate(item) for item in payload]

    assert cases
    assert len({case.case_id for case in cases}) == len(cases)
    assert any(case.safety_critical for case in cases)
    with pytest.raises(ValidationError):
        cases[0].description = "Modified between baseline and candidate runs"
