from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from distill.schema import (
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
