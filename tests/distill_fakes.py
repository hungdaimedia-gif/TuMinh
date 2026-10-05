from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from distill.schema import (
    BenchmarkMetadata,
    EvalCase,
    ImportProvenance,
    ObservableEvent,
    TrajectoryExport,
)


def now_utc():
    return datetime.now(UTC)


def valid_trajectory(
    outcome: Literal["succeeded", "failed"] = "succeeded",
    verification_passed: bool = True,
    eval_passed: bool = True,
) -> TrajectoryExport:
    if outcome == "succeeded" and not verification_passed:
        outcome = "failed"
    return TrajectoryExport(
        schema_version=1,
        trajectory_id="trajectory-synthetic-1",
        task_id="task-synthetic-1",
        source_type="runtime",
        provenance=ImportProvenance(
            source_name="synthetic-runtime",
            imported_at=now_utc(),
            source_sha256="c" * 64,
        ),
        task_outcome=outcome,
        verification_passed=verification_passed,
        eval_passed=eval_passed,
        events=[
            ObservableEvent(
                schema_version=1,
                task_id="task-synthetic-1",
                sequence_number=1,
                event_type="task",
                timestamp=now_utc(),
                data={"title": "Synthetic task"},
            ),
            ObservableEvent(
                schema_version=1,
                task_id="task-synthetic-1",
                sequence_number=2,
                event_type="verification",
                timestamp=now_utc(),
                data={"passed": verification_passed},
            ),
        ],
    )


def synthetic_eval_cases() -> list[EvalCase]:
    fixture = Path(__file__).parents[1] / "distill" / "evals" / "cases.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    return [EvalCase.model_validate(item) for item in payload]


def complete_benchmark_metadata() -> BenchmarkMetadata:
    return BenchmarkMetadata(
        benchmark_id="benchmark-synthetic-1",
        baseline_model_id="baseline-model",
        baseline_model_revision="baseline-model-revision",
        candidate_model_id="candidate-model",
        candidate_model_revision="candidate-model-revision",
        dataset_hash="d" * 64,
        eval_set_version="tuminh-core-v1",
        safety_critical_case_ids=[
            case.case_id for case in synthetic_eval_cases() if case.safety_critical
        ],
        tool_limits={"max_tool_calls": 8, "timeout_seconds": 30},
        decoding_config={"temperature": 0, "max_output_length": 4096},
        software_metadata={"tuminh": "0.1.0", "python": "3.11"},
        hardware_metadata={"device": "cpu", "processor": "synthetic"},
    )
