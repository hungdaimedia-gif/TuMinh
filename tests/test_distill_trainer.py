import hashlib
import json

import pytest
from pydantic import ValidationError

from distill.schema import TrainingRow
from distill.trainer import (
    TrainingConfig,
    TrainingDatasetError,
    TrainingEnvironmentError,
    TrainingError,
    TrainingOutOfMemoryError,
    Trainer,
)
from tests.distill_fakes import FakeTrainingBackend, valid_training_config


class FixedHardwareProbe:
    def __init__(self, peak_vram: int | None):
        self.peak_vram = peak_vram

    def measure_peak_vram(self) -> int | None:
        return self.peak_vram


def write_dataset(path, *, verified=True, evaluated=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = TrainingRow(
        task_id="synthetic-task",
        messages=[{"role": "user", "content": "Synthetic prompt"}],
        verification_passed=verified,
        eval_passed=evaluated,
        provenance_id="a" * 64,
    )
    path.write_text(row.model_dump_json() + "\n", encoding="utf-8")
    return path


def make_trainer(tmp_path, backend, peak_vram=123456):
    return Trainer(
        backend=backend,
        candidate_root=tmp_path / "candidates",
        hardware_probe=FixedHardwareProbe(peak_vram),
    )


def test_training_config_rejects_invalid_bounds_and_missing_metadata(tmp_path):
    config = valid_training_config(tmp_path)
    payload = config.model_dump()

    for field, value in (
        ("sequence_length", 0),
        ("sequence_length", 8193),
        ("batch_size", 0),
        ("lora_rank", 0),
        ("lora_rank", 257),
        ("gradient_accumulation_steps", 0),
        ("model_id", ""),
        ("model_revision", ""),
        ("license_or_authorization", ""),
    ):
        with pytest.raises(ValidationError):
            TrainingConfig(**{**payload, field: value})


def test_training_config_loads_conservative_toml_relative_to_file(tmp_path):
    config_file = tmp_path / "training.toml"
    config_file.write_text(
        """
[training]
model_id = "synthetic/model"
model_revision = "revision-sha"
license_or_authorization = "Authorized synthetic test model"
quantization = "4bit"
sequence_length = 1024
batch_size = 1
gradient_accumulation_steps = 8
lora_rank = 8
output_path = "candidates/run-from-toml"
hardware_profile = "rtx_3060_ti_8gb"
""",
        encoding="utf-8",
    )

    config = TrainingConfig.from_toml(config_file)

    assert config.output_path == tmp_path / "candidates" / "run-from-toml"
    assert config.sequence_length == 1024
    assert config.hardware_profile == "rtx_3060_ti_8gb"


def test_trainer_calls_fake_backend_and_returns_measured_candidate_manifest(tmp_path):
    backend = FakeTrainingBackend(cuda_available=True)
    trainer = make_trainer(tmp_path, backend)
    dataset = write_dataset(tmp_path / "curated" / "dataset.jsonl")
    config = valid_training_config(tmp_path)

    manifest = trainer.train(dataset, config)

    assert backend.invocations == 1
    assert manifest.candidate_id == config.output_path.name
    assert manifest.model_id == config.model_id
    assert manifest.model_revision == config.model_revision
    assert manifest.base_model_license == config.license_or_authorization
    assert manifest.dataset_hash == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert manifest.benchmark_id.startswith("training-reserved-")
    assert manifest.runtime_metadata["training_completed"] is True
    assert manifest.hardware_metadata["peak_vram_bytes"] == 123456
    assert (config.output_path / "adapter-config.json").is_file()
    assert backend.dataset_paths[0] != dataset
    assert backend.dataset_paths[0].parent.parent == tmp_path / "candidates"
    assert not (tmp_path / "registry.json").exists()


def test_each_candidate_run_gets_a_distinct_reserved_benchmark_id(tmp_path):
    backend = FakeTrainingBackend(cuda_available=True)
    trainer = make_trainer(tmp_path, backend, peak_vram=None)
    dataset = write_dataset(tmp_path / "curated" / "dataset.jsonl")
    first_config = valid_training_config(tmp_path)
    second_config = first_config.model_copy(
        update={"output_path": tmp_path / "candidates" / "candidate-synthetic-2"}
    )

    first = trainer.train(dataset, first_config)
    second = trainer.train(dataset, second_config)

    assert first.benchmark_id != second.benchmark_id
    assert first.hardware_metadata["peak_vram_bytes"] is None
    assert backend.invocations == 2


def test_trainer_rejects_missing_dataset_before_backend_invocation(tmp_path):
    backend = FakeTrainingBackend(cuda_available=True)
    trainer = make_trainer(tmp_path, backend)

    with pytest.raises(TrainingDatasetError, match="does not exist"):
        trainer.train(tmp_path / "missing.jsonl", valid_training_config(tmp_path))

    assert backend.invocations == 0


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "empty"),
        ("not json\n", "invalid"),
        (
            json.dumps(
                {
                    "task_id": "unverified",
                    "messages": [{"role": "user", "content": "prompt"}],
                    "verification_passed": False,
                    "eval_passed": True,
                    "provenance_id": "a" * 64,
                }
            )
            + "\n",
            "verification",
        ),
        (
            json.dumps(
                {
                    "task_id": "unevaluated",
                    "messages": [{"role": "user", "content": "prompt"}],
                    "verification_passed": True,
                    "eval_passed": False,
                    "provenance_id": "a" * 64,
                }
            )
            + "\n",
            "eval",
        ),
        (
            json.dumps(
                {
                    "task_id": "coerced-verification",
                    "messages": [{"role": "user", "content": "prompt"}],
                    "verification_passed": 1,
                    "eval_passed": True,
                    "provenance_id": "a" * 64,
                }
            )
            + "\n",
            "invalid",
        ),
    ],
)
def test_trainer_validates_entire_dataset_before_backend(
    tmp_path, content, message
):
    dataset = tmp_path / "curated" / "dataset.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(content, encoding="utf-8")
    backend = FakeTrainingBackend(cuda_available=True)
    trainer = make_trainer(tmp_path, backend)

    with pytest.raises(TrainingDatasetError, match=message):
        trainer.train(dataset, valid_training_config(tmp_path))

    assert backend.invocations == 0
    assert not (tmp_path / "candidates").exists()


def test_trainer_reports_missing_cuda_without_registry_changes(tmp_path):
    backend = FakeTrainingBackend(cuda_available=False)
    trainer = make_trainer(tmp_path, backend)
    dataset = write_dataset(tmp_path / "curated" / "dataset.jsonl")

    with pytest.raises(TrainingEnvironmentError, match="CUDA"):
        trainer.train(dataset, valid_training_config(tmp_path))

    assert backend.invocations == 0
    assert not (tmp_path / "registry.json").exists()
    assert not (tmp_path / "candidates").exists()


def test_trainer_rejects_outputs_outside_candidate_root(tmp_path):
    backend = FakeTrainingBackend(cuda_available=True)
    trainer = make_trainer(tmp_path, backend)
    dataset = write_dataset(tmp_path / "curated" / "dataset.jsonl")
    config = valid_training_config(tmp_path).model_copy(
        update={"output_path": tmp_path / "outside" / "candidate-test"}
    )

    with pytest.raises(TrainingError, match="candidate"):
        trainer.train(dataset, config)

    assert backend.invocations == 0
    assert not (tmp_path / "outside").exists()


def test_rtx_3060_ti_profile_rejects_aggressive_settings(tmp_path):
    backend = FakeTrainingBackend(cuda_available=True)
    trainer = make_trainer(tmp_path, backend)
    dataset = write_dataset(tmp_path / "curated" / "dataset.jsonl")
    config = valid_training_config(tmp_path).model_copy(
        update={"sequence_length": 4096, "hardware_profile": "rtx_3060_ti_8gb"}
    )

    with pytest.raises(TrainingOutOfMemoryError, match="sequence_length"):
        trainer.train(dataset, config)

    assert backend.invocations == 0


def test_backend_failure_preserves_candidate_and_registry_state(tmp_path):
    backend = FakeTrainingBackend(
        cuda_available=True,
        failure=RuntimeError("backend unavailable"),
    )
    trainer = make_trainer(tmp_path, backend)
    dataset = write_dataset(tmp_path / "curated" / "dataset.jsonl")
    registry = tmp_path / "registry.json"
    registry.write_text('{"active_model":"baseline"}\n', encoding="utf-8")

    with pytest.raises(TrainingError, match="Training backend failed"):
        trainer.train(dataset, valid_training_config(tmp_path))

    assert registry.read_text(encoding="utf-8") == '{"active_model":"baseline"}\n'
    assert not valid_training_config(tmp_path).output_path.exists()
    assert list((tmp_path / "candidates").iterdir()) == []


def test_backend_out_of_memory_is_reported_as_typed_error_and_cleans_staging(tmp_path):
    backend = FakeTrainingBackend(
        cuda_available=True,
        failure=RuntimeError("CUDA out of memory"),
    )
    trainer = make_trainer(tmp_path, backend)
    dataset = write_dataset(tmp_path / "curated" / "dataset.jsonl")

    with pytest.raises(TrainingOutOfMemoryError, match="lower sequence_length"):
        trainer.train(dataset, valid_training_config(tmp_path))

    assert list((tmp_path / "candidates").iterdir()) == []


def test_unknown_quantization_and_hardware_profile_are_rejected(tmp_path):
    config = valid_training_config(tmp_path)
    payload = config.model_dump()

    with pytest.raises(ValidationError, match="quantization"):
        TrainingConfig(**{**payload, "quantization": "magic"})
    with pytest.raises(ValidationError, match="hardware_profile"):
        TrainingConfig(**{**payload, "hardware_profile": "fictional-gpu"})
