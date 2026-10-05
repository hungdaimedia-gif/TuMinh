from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from distill.deploy import ConversionConfig, OllamaError
from distill.schema import (
    BenchmarkReport,
    BenchmarkMetadata,
    CandidateManifest,
    EvalCase,
    ImportProvenance,
    ModelScore,
    ObservableEvent,
    TrajectoryExport,
)
from distill.trainer import TrainingConfig


def now_utc():
    return datetime.now(UTC)


def valid_training_config(output_root: Path) -> TrainingConfig:
    return TrainingConfig(
        model_id="synthetic/model",
        model_revision="synthetic-revision-sha",
        license_or_authorization="Authorized synthetic test model",
        quantization="4bit",
        sequence_length=1024,
        batch_size=1,
        gradient_accumulation_steps=8,
        lora_rank=8,
        output_path=Path(output_root) / "candidates" / "candidate-synthetic-1",
        hardware_profile=None,
    )


class FakeTrainingBackend:
    def __init__(self, cuda_available: bool, failure: Exception | None = None):
        self.cuda_available = cuda_available
        self.failure = failure
        self.invocations = 0
        self.dataset_paths: list[Path] = []

    def train(self, dataset_path: Path, config: TrainingConfig) -> dict[str, object]:
        self.invocations += 1
        self.dataset_paths.append(dataset_path)
        if self.failure is not None:
            raise self.failure
        config.output_path.mkdir(parents=True, exist_ok=True)
        (config.output_path / "adapter-config.json").write_text(
            json.dumps({"test_backend": True}),
            encoding="utf-8",
        )
        return {
            "backend": "synthetic-fake",
            "training_completed": True,
            "dataset_path": dataset_path.name,
        }


class FakeConverter:
    def __init__(self, failure: Exception | None = None):
        self.failure = failure
        self.calls: list[tuple[Path, Path, ConversionConfig]] = []

    def convert(
        self,
        adapter_path: Path,
        output_path: Path,
        config: ConversionConfig,
    ) -> Path:
        self.calls.append((adapter_path, output_path, config))
        output_path.mkdir(parents=True, exist_ok=True)
        partial_output = output_path / "partial.gguf"
        partial_output.write_bytes(b"synthetic-conversion")
        if self.failure is not None:
            raise self.failure
        model_path = output_path / "model.gguf"
        partial_output.replace(model_path)
        return model_path


class FakeOllamaAdapter:
    def __init__(
        self,
        install_error: OllamaError | None = None,
        *,
        confirm_installed: bool = True,
        returned_tag: str | None = None,
    ):
        self.install_error = install_error
        self.confirm_installed = confirm_installed
        self.returned_tag = returned_tag
        self.installed_tags: list[str] = []
        self.install_calls: list[tuple[Path, str]] = []

    def install_candidate(self, model_path: Path, candidate_id: str) -> str:
        self.install_calls.append((model_path, candidate_id))
        if self.install_error is not None:
            raise self.install_error
        self.installed_tags.append(candidate_id)
        return self.returned_tag if self.returned_tag is not None else candidate_id

    def is_installed(self, model_tag: str) -> bool:
        return self.confirm_installed and model_tag in self.installed_tags


def valid_candidate_manifest(candidate_id: str) -> CandidateManifest:
    return CandidateManifest(
        candidate_id=candidate_id,
        model_id="candidate-model",
        model_revision="candidate-model-revision",
        base_model_license="synthetic-license",
        dataset_hash="d" * 64,
        training_config={"rank": 4},
        runtime_metadata={"format": "synthetic"},
        hardware_metadata={"device": "cpu"},
        benchmark_id="benchmark-synthetic-1",
    )


def ineligible_benchmark_report() -> BenchmarkReport:
    metadata = complete_benchmark_metadata()
    cases = synthetic_eval_cases()
    case_results = {case.case_id: True for case in cases}
    safety_case_ids = metadata.safety_critical_case_ids
    safety_results = {case_id: True for case_id in safety_case_ids}
    failed_safety_case = safety_case_ids[0]
    safety_results[failed_safety_case] = False
    candidate_results = case_results.copy()
    candidate_results[failed_safety_case] = False
    baseline_score = ModelScore(
        model_id=metadata.baseline_model_id,
        model_revision=metadata.baseline_model_revision,
        case_results=case_results,
        passed_count=len(case_results),
        total_count=len(case_results),
        pass_rate=1,
        resource_measurements={"duration_seconds": 1},
    )
    candidate_score = ModelScore(
        model_id=metadata.candidate_model_id,
        model_revision=metadata.candidate_model_revision,
        case_results=candidate_results,
        passed_count=sum(candidate_results.values()),
        total_count=len(candidate_results),
        pass_rate=sum(candidate_results.values()) / len(candidate_results),
        resource_measurements={"duration_seconds": 1},
    )
    return BenchmarkReport(
        metadata=metadata,
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        eval_set_hash="e" * 64,
        eval_case_ids=list(case_results),
        safety_case_ids=safety_case_ids,
        safety_results=safety_results,
    )


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
