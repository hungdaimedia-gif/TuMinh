from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import distill.cli as cli
from distill.registry import DistillModelRegistry
from distill.schema import (
    BenchmarkReport,
    CandidateManifest,
    ModelScore,
)
from tests.distill_fakes import (
    FakeConverter,
    FakeOllamaAdapter,
    FakeTrainingBackend,
    complete_benchmark_metadata,
    synthetic_eval_cases,
    valid_candidate_manifest,
    valid_trajectory,
)


def make_report(
    candidate: CandidateManifest,
    *,
    fail_safety_case: bool = False,
) -> BenchmarkReport:
    metadata = complete_benchmark_metadata().model_copy(
        update={
            "benchmark_id": candidate.benchmark_id,
            "candidate_model_id": candidate.model_id,
            "candidate_model_revision": candidate.model_revision,
            "dataset_hash": candidate.dataset_hash,
        }
    )
    results = {case.case_id: True for case in synthetic_eval_cases()}
    candidate_results = results.copy()
    safety_results = {
        case_id: True for case_id in metadata.safety_critical_case_ids
    }
    if fail_safety_case:
        failed_case_id = metadata.safety_critical_case_ids[0]
        candidate_results[failed_case_id] = False
        safety_results[failed_case_id] = False
    baseline = ModelScore(
        model_id=metadata.baseline_model_id,
        model_revision=metadata.baseline_model_revision,
        case_results=results,
        passed_count=len(results),
        total_count=len(results),
        pass_rate=1,
        resource_measurements={"duration_seconds": 1},
    )
    candidate_score = ModelScore(
        model_id=candidate.model_id,
        model_revision=candidate.model_revision,
        case_results=candidate_results,
        passed_count=sum(candidate_results.values()),
        total_count=len(results),
        pass_rate=sum(candidate_results.values()) / len(results),
        resource_measurements={"duration_seconds": 1},
    )
    return BenchmarkReport(
        metadata=metadata,
        baseline_score=baseline,
        candidate_score=candidate_score,
        eval_set_hash="e" * 64,
        eval_case_ids=list(results),
        safety_case_ids=metadata.safety_critical_case_ids,
        safety_results=safety_results,
    )


class FixedEvaluator:
    def __init__(self, *, failure: bool = False):
        self.failure = failure
        self.calls = []

    def evaluate(self, model_name, cases, *, tool_limits, decoding_config):
        self.calls.append(model_name)
        if self.failure:
            raise RuntimeError("evaluation failed")
        return ModelScore(
            model_id=(
                "baseline-model"
                if model_name == "baseline-tag"
                else "candidate-model"
            ),
            model_revision=(
                "baseline-model-revision"
                if model_name == "baseline-tag"
                else "candidate-model-revision"
            ),
            case_results={case.case_id: True for case in cases},
            passed_count=len(cases),
            total_count=len(cases),
            pass_rate=1,
            resource_measurements={"duration_seconds": 1},
        )


def test_help_lists_all_distillation_commands(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    for command in (
        "export-runtime",
        "import-external",
        "curate",
        "train",
        "deploy",
        "benchmark",
        "init-active",
        "promote",
        "rollback",
        "purge",
    ):
        assert command in help_text


@pytest.mark.parametrize(
    "command",
    (
        "export-runtime",
        "import-external",
        "curate",
        "train",
        "deploy",
        "benchmark",
        "init-active",
        "promote",
        "rollback",
        "purge",
    ),
)
def test_each_command_help_is_offline_and_does_not_load_plugins(
    command, capsys, monkeypatch
):
    def reject_plugin_import(*_args, **_kwargs):
        raise AssertionError("help must not load a local evaluator")

    monkeypatch.setattr(cli.importlib, "import_module", reject_plugin_import)
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--help"])

    assert error.value.code == 0
    assert command in capsys.readouterr().out


def test_import_external_requires_authorization_provenance(capsys, tmp_path):
    source = tmp_path / "external.jsonl"
    source.write_text("{}\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        cli.main(["import-external", str(source), "--output", str(tmp_path / "out.jsonl")])

    assert error.value.code == 2
    assert not (tmp_path / "out.jsonl").exists()
    assert "required" in capsys.readouterr().err


def test_curate_redacts_model_api_key_and_writes_dataset_manifest(
    capsys, monkeypatch, tmp_path
):
    api_key = "sk-local-test-secret"
    monkeypatch.setenv("MODEL_API_KEY", api_key)
    source = tmp_path / "trajectory.jsonl"
    trajectory = valid_trajectory()
    task_event = trajectory.events[0].model_copy(
        update={"data": {"title": f"task uses {api_key}"}}
    )
    trajectory = trajectory.model_copy(update={"events": [task_event, *trajectory.events[1:]]})
    cli.write_trajectory_exports([trajectory], source)

    output = tmp_path / "curated.jsonl"
    result = cli.main(
        ["curate", "--input", str(source), "--output", str(output)]
    )

    assert result == 0
    assert api_key not in output.read_text(encoding="utf-8")
    assert "[REDACTED]" in output.read_text(encoding="utf-8")
    assert output.with_suffix(".manifest.json").is_file()
    assert "Curated" in capsys.readouterr().out


def test_train_invokes_trainer_and_does_not_deploy(
    capsys, monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "task_id": "task",
                "messages": [{"role": "user", "content": "do work"}],
                "verification_passed": True,
                "eval_passed": True,
                "provenance_id": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "training.toml"
    config.write_text(
        f"""[training]
model_id = "local/model"
model_revision = "revision-1"
license_or_authorization = "authorized"
quantization = "4bit"
sequence_length = 512
batch_size = 1
gradient_accumulation_steps = 1
lora_rank = 4
output_path = "{(tmp_path / 'candidates' / 'candidate-cli').as_posix()}"
""",
        encoding="utf-8",
    )

    class NoOpHardwareProbe:
        def reset_peak_measurement(self):
            pass

        def measure_peak_vram(self):
            return None

    def reject_real_hardware_probe():
        pytest.fail("CLI test must not construct the optional torch hardware probe")

    monkeypatch.setattr(
        cli, "OptionalTorchHardwareProbe", reject_real_hardware_probe
    )
    hardware_probe = NoOpHardwareProbe()
    trainer = cli.Trainer(
        backend=FakeTrainingBackend(cuda_available=True),
        candidate_root=tmp_path / "candidates",
        hardware_probe=hardware_probe,
    )
    result = cli.main(
        [
            "train",
            "--config",
            str(config),
            "--dataset",
            str(dataset),
        ],
        services=cli.DistillServices(trainer=trainer),
    )

    assert result == 0
    manifest_path = (
        tmp_path / "candidates" / "candidate-cli" / "candidate-manifest.json"
    )
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["hardware_metadata"]["peak_vram_bytes"] is None
    assert "candidate-manifest.json" in capsys.readouterr().out


def test_benchmark_uses_reserved_id_and_fails_closed_on_evaluator_error(
    capsys, tmp_path
):
    candidate = valid_candidate_manifest("candidate-benchmark")
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active(
        "baseline-tag",
        baseline_model_id="baseline-model",
        baseline_model_revision="baseline-model-revision",
    )
    registry.register_candidate(candidate)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(candidate.model_dump_json(), encoding="utf-8")
    output = tmp_path / "benchmark.json"
    evaluator = FixedEvaluator(failure=True)

    result = cli.main(
        [
            "benchmark",
            "--candidate-manifest",
            str(candidate_path),
            "--registry",
            str(registry.path),
            "--output",
            str(output),
            "--evaluator",
            "tests.distill_fakes:FakeEvaluator",
        ],
        services=cli.DistillServices(
            registry=registry,
            evaluator_factory=lambda _spec: evaluator,
        ),
    )

    assert result == 2
    assert evaluator.calls == ["baseline-tag"]
    assert not output.exists()
    assert "failed" in capsys.readouterr().err.lower()


def test_benchmark_writes_typed_report_with_reserved_benchmark_id(tmp_path):
    candidate = valid_candidate_manifest("candidate-benchmark")
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active(
        "baseline-tag",
        baseline_model_id="baseline-model",
        baseline_model_revision="baseline-model-revision",
    )
    registry.register_candidate(candidate)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(candidate.model_dump_json(), encoding="utf-8")
    output = tmp_path / "benchmark.json"
    evaluator = FixedEvaluator()

    result = cli.main(
        [
            "benchmark",
            "--candidate-manifest",
            str(candidate_path),
            "--registry",
            str(registry.path),
            "--output",
            str(output),
            "--evaluator",
            "tests.distill_fakes:FakeEvaluator",
        ],
        services=cli.DistillServices(
            registry=registry,
            evaluator_factory=lambda _spec: evaluator,
        ),
    )

    assert result == 0
    report = BenchmarkReport.model_validate_json(output.read_text(encoding="utf-8"))
    assert report.metadata.benchmark_id == candidate.benchmark_id
    assert report.metadata.candidate_model_id == candidate.model_id
    assert report.eligible_for_promotion
    assert evaluator.calls == ["baseline-tag", candidate.candidate_id]


def test_negative_promotion_confirmation_preserves_active_pointer(capsys, tmp_path):
    candidate = valid_candidate_manifest("candidate-promote")
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active(
        "baseline-v1",
        baseline_model_id="baseline-model",
        baseline_model_revision="baseline-model-revision",
    )
    registry.register_candidate(candidate)
    report_path = tmp_path / "report.json"
    report_path.write_text(make_report(candidate).model_dump_json(), encoding="utf-8")

    result = cli.main(
        [
            "promote",
            "--candidate-id",
            candidate.candidate_id,
            "--report",
            str(report_path),
            "--registry",
            str(registry.path),
        ],
        services=cli.DistillServices(registry=registry),
        input_fn=lambda _prompt: "no",
    )

    assert result == 0
    assert registry.read().active_model == "baseline-v1"
    assert registry.read().previous_model is None
    assert "cancel" in capsys.readouterr().out.lower()


def test_positive_promotion_requires_named_operator_and_records_approval(tmp_path):
    candidate = valid_candidate_manifest("candidate-promote")
    class RecordingRegistry(DistillModelRegistry):
        approval = None

        def promote(self, candidate_id, report, approval):
            self.approval = approval
            return super().promote(candidate_id, report, approval)

    registry = RecordingRegistry(tmp_path / "registry.json")
    registry.initialize_active(
        "baseline-v1",
        baseline_model_id="baseline-model",
        baseline_model_revision="baseline-model-revision",
    )
    registry.register_candidate(candidate)
    report_path = tmp_path / "report.json"
    report_path.write_text(make_report(candidate).model_dump_json(), encoding="utf-8")
    answers = iter(["yes", "operator@example.test"])

    result = cli.main(
        [
            "promote",
            "--candidate-id",
            candidate.candidate_id,
            "--report",
            str(report_path),
            "--registry",
            str(registry.path),
        ],
        services=cli.DistillServices(registry=registry),
        input_fn=lambda _prompt: next(answers),
    )

    assert result == 0
    assert registry.read().active_model == candidate.candidate_id
    assert registry.read().promoted_at.tzinfo is not None
    assert registry.approval.approved_by == "operator@example.test"


def test_ineligible_benchmark_report_is_refused_even_after_confirmation(
    capsys, tmp_path
):
    candidate = valid_candidate_manifest("candidate-promote")
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active(
        "baseline-v1",
        baseline_model_id="baseline-model",
        baseline_model_revision="baseline-model-revision",
    )
    registry.register_candidate(candidate)
    report_path = tmp_path / "report.json"
    ineligible_report = make_report(candidate, fail_safety_case=True)
    assert not ineligible_report.eligible_for_promotion
    report_path.write_text(ineligible_report.model_dump_json(), encoding="utf-8")
    answers = iter(["yes", "operator@example.test"])

    result = cli.main(
        [
            "promote",
            "--candidate-id",
            candidate.candidate_id,
            "--report",
            str(report_path),
            "--registry",
            str(registry.path),
        ],
        services=cli.DistillServices(registry=registry),
        input_fn=lambda _prompt: next(answers),
    )

    assert result == 2
    assert registry.read().active_model == "baseline-v1"
    assert registry.read().previous_model is None
    captured = capsys.readouterr()
    assert "safety" in captured.out.lower()
    assert "eligible" in captured.err.lower()


def test_rollback_command_restores_previous_active_model(capsys, tmp_path):
    candidate = valid_candidate_manifest("candidate-promote")
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active(
        "baseline-v1",
        baseline_model_id="baseline-model",
        baseline_model_revision="baseline-model-revision",
    )
    registry.register_candidate(candidate)
    registry.promote(
        candidate.candidate_id,
        make_report(candidate),
        cli.HumanApproval(approved_by="operator", confirmed_at=datetime.now(UTC)),
    )

    result = cli.main(
        ["rollback", "--registry", str(registry.path)],
        services=cli.DistillServices(registry=registry),
    )

    assert result == 0
    assert registry.resolve_active_model_name("tuminh-active") == "baseline-v1"
    assert "baseline-v1" in capsys.readouterr().out


def test_init_active_requires_and_persists_exact_baseline_identity(tmp_path):
    registry = DistillModelRegistry(tmp_path / "registry.json")

    result = cli.main(
        [
            "init-active",
            "--registry",
            str(registry.path),
            "--model-tag",
            "baseline-v1",
            "--baseline-model-id",
            "base/model",
            "--baseline-model-revision",
            "revision-sha",
        ],
        services=cli.DistillServices(registry=registry),
    )

    assert result == 0
    state = registry.read()
    assert state.active_model == "baseline-v1"
    assert state.initialized_baseline_model_id == "base/model"
    assert state.initialized_baseline_revision == "revision-sha"


def test_cli_registry_default_uses_configured_dotenv_path(tmp_path, monkeypatch):
    registry_path = tmp_path / "private-registry" / "registry.json"
    (tmp_path / ".env").write_text(
        f"WORKSPACE_ROOT={tmp_path}\n"
        "DISTILL_REGISTRY_PATH=private-registry/registry.json\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = cli.main(
        [
            "init-active",
            "--model-tag",
            "baseline-v1",
            "--baseline-model-id",
            "base/model",
            "--baseline-model-revision",
            "revision-sha",
        ]
    )

    assert result == 0
    assert DistillModelRegistry(registry_path).read().active_model == "baseline-v1"


def test_deploy_registers_candidate_only_after_deploy_service_succeeds(tmp_path):
    candidate = valid_candidate_manifest("candidate-deploy-cli")
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active("baseline-v1")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter.bin").write_bytes(b"adapter")
    (adapter / "candidate-manifest.json").write_text(
        candidate.model_dump_json(), encoding="utf-8"
    )
    deployer = cli.CandidateDeployer(
        converter=FakeConverter(),
        ollama=FakeOllamaAdapter(),
        registry=registry,
        candidate_root=tmp_path / "candidates",
    )
    manifest_path = tmp_path / "candidate.json"
    manifest_path.write_text(candidate.model_dump_json(), encoding="utf-8")

    result = cli.main(
        [
            "deploy",
            "--manifest",
            str(manifest_path),
            "--adapter",
            str(adapter),
            "--registry",
            str(registry.path),
        ],
        services=cli.DistillServices(registry=registry, deployer=deployer),
    )

    assert result == 0
    assert candidate.candidate_id in registry.read().candidates
    assert registry.read().active_model == "baseline-v1"


def test_purge_uses_exact_source_hash(capsys, tmp_path):
    source_bytes = b"authorized raw source"
    import hashlib

    source_hash = hashlib.sha256(source_bytes).hexdigest()
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / f"{source_hash}.jsonl").write_bytes(source_bytes)

    result = cli.main(
        [
            "purge",
            "--source-sha256",
            source_hash,
            "--raw-root",
            str(raw_root),
            "--curated-root",
            str(tmp_path / "datasets"),
        ]
    )

    assert result == 0
    assert not (raw_root / f"{source_hash}.jsonl").exists()
    assert "1 raw" in capsys.readouterr().out


def test_cli_reports_invalid_input_without_claiming_success(capsys, tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")

    result = cli.main(
        [
            "promote",
            "--candidate-id",
            "candidate",
            "--report",
            str(invalid),
            "--registry",
            str(tmp_path / "registry.json"),
        ],
        services=cli.DistillServices(
            registry=DistillModelRegistry(tmp_path / "registry.json")
        ),
    )

    assert result == 2
    assert "promot" not in capsys.readouterr().out.lower()
