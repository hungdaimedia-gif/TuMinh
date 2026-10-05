from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Protocol

from pydantic import ValidationError

from distill.schema import (
    BenchmarkMetadata,
    BenchmarkReport,
    EvalCase,
    ModelScore,
)


class ModelEvaluator(Protocol):
    def evaluate(self, model_name: str, cases: Sequence[EvalCase]) -> ModelScore: ...


class BenchmarkError(ValueError):
    """Raised when a model cannot be evaluated into trustworthy benchmark evidence."""


class BenchmarkRunner:
    def __init__(
        self,
        evaluator: ModelEvaluator,
        eval_set: Sequence[EvalCase],
    ):
        self.evaluator = evaluator
        try:
            self.eval_set = tuple(EvalCase.model_validate(case) for case in eval_set)
        except (ValidationError, ValueError) as error:
            raise BenchmarkError("Evaluation set contains an invalid case") from error
        case_ids = [case.case_id for case in self.eval_set]
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise BenchmarkError("Evaluation set must contain unique cases")

    def compare(
        self,
        baseline: str,
        candidate: str,
        metadata: BenchmarkMetadata,
    ) -> BenchmarkReport:
        if not baseline.strip() or not candidate.strip() or baseline == candidate:
            raise BenchmarkError("Baseline and candidate must be distinct model names")
        try:
            validated_metadata = BenchmarkMetadata.model_validate(metadata).model_copy(deep=True)
        except (ValidationError, ValueError) as error:
            raise BenchmarkError("Benchmark metadata is invalid") from error

        baseline_score = self._evaluate(baseline)
        candidate_score = self._evaluate(candidate)
        case_ids = [case.case_id for case in self.eval_set]
        safety_case_ids = [
            case.case_id for case in self.eval_set if case.safety_critical
        ]
        safety_results = {
            case_id: candidate_score.case_results[case_id]
            for case_id in safety_case_ids
            if case_id in candidate_score.case_results
        }
        canonical_cases = json.dumps(
            [case.model_dump(mode="json") for case in self.eval_set],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return BenchmarkReport(
            metadata=validated_metadata,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            eval_set_hash=hashlib.sha256(canonical_cases).hexdigest(),
            eval_case_ids=case_ids,
            safety_case_ids=safety_case_ids,
            safety_results=safety_results,
        )

    def _evaluate(self, model_name: str) -> ModelScore:
        try:
            result = self.evaluator.evaluate(model_name, self.eval_set)
            return ModelScore.model_validate(result)
        except Exception as error:
            raise BenchmarkError(f"Model evaluation failed for {model_name!r}") from error


def is_candidate_eligible(
    baseline_pass_rate: float,
    candidate_pass_rate: float,
    safety_results: Mapping[str, bool],
    metadata: BenchmarkMetadata,
) -> bool:
    try:
        validated_metadata = BenchmarkMetadata.model_validate(metadata)
    except (ValidationError, ValueError):
        return False
    if (
        not validated_metadata.complete
        or not safety_results
        or set(safety_results) != set(validated_metadata.safety_critical_case_ids)
        or not all(isinstance(passed, bool) and passed for passed in safety_results.values())
    ):
        return False
    if (
        isinstance(baseline_pass_rate, bool)
        or isinstance(candidate_pass_rate, bool)
        or not isinstance(baseline_pass_rate, (float, int))
        or not isinstance(candidate_pass_rate, (float, int))
        or not isfinite(baseline_pass_rate)
        or not isfinite(candidate_pass_rate)
    ):
        return False
    return (
        0 <= baseline_pass_rate <= 1
        and 0 <= candidate_pass_rate <= 1
        and candidate_pass_rate >= baseline_pass_rate
    )
