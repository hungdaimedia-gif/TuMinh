from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import distill.registry as registry_module
from distill.registry import (
    DistillModelRegistry,
    HumanApproval,
    PromotionError,
    RegistryError,
)
from tests.distill_fakes import (
    complete_benchmark_metadata,
    ineligible_benchmark_report,
    now_utc,
    synthetic_eval_cases,
    valid_candidate_manifest,
)
from distill.schema import BenchmarkReport, ModelScore


def eligible_report() -> BenchmarkReport:
    metadata = complete_benchmark_metadata()
    case_results = {case.case_id: True for case in synthetic_eval_cases()}
    baseline = ModelScore(
        model_id=metadata.baseline_model_id,
        model_revision=metadata.baseline_model_revision,
        case_results=case_results,
        passed_count=len(case_results),
        total_count=len(case_results),
        pass_rate=1,
        resource_measurements={"duration_seconds": 1},
    )
    candidate = ModelScore(
        model_id=metadata.candidate_model_id,
        model_revision=metadata.candidate_model_revision,
        case_results=case_results,
        passed_count=len(case_results),
        total_count=len(case_results),
        pass_rate=1,
        resource_measurements={"duration_seconds": 1},
    )
    return BenchmarkReport(
        metadata=metadata,
        baseline_score=baseline,
        candidate_score=candidate,
        eval_set_hash="e" * 64,
        eval_case_ids=list(case_results),
        safety_case_ids=metadata.safety_critical_case_ids,
        safety_results={case_id: True for case_id in metadata.safety_critical_case_ids},
    )


def setup_registry(tmp_path, candidate_id="candidate-v2"):
    registry = DistillModelRegistry(tmp_path / "registry-matched.json")
    registry.initialize_active(
        "baseline-v1",
        baseline_model_id="baseline-model",
        baseline_model_revision="baseline-model-revision",
    )
    registry.register_candidate(valid_candidate_manifest(candidate_id))
    return registry


def approval():
    return HumanApproval(approved_by="operator", confirmed_at=now_utc())


def test_promotion_rejects_missing_or_mismatched_initialized_baseline_identity(tmp_path):
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active("baseline-v1")
    registry.register_candidate(valid_candidate_manifest("candidate-v2"))

    with pytest.raises(PromotionError, match="baseline"):
        registry.promote(
            "candidate-v2",
            report=eligible_report(),
            approval=approval(),
        )

    registry = DistillModelRegistry(tmp_path / "registry-matched.json")
    registry.initialize_active(
        "baseline-v1",
        baseline_model_id="baseline-model",
        baseline_model_revision="baseline-model-revision",
    )
    registry.register_candidate(valid_candidate_manifest("candidate-v2"))
    mismatched = eligible_report().model_copy(
        update={
            "metadata": eligible_report().metadata.model_copy(
                update={"baseline_model_revision": "different-baseline-revision"}
            ),
            "baseline_score": eligible_report().baseline_score.model_copy(
                update={"model_revision": "different-baseline-revision"}
            ),
        }
    )

    with pytest.raises(PromotionError, match="baseline"):
        registry.promote("candidate-v2", report=mismatched, approval=approval())


def test_promotion_accepts_matching_initialized_baseline_identity(tmp_path):
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active(
        "baseline-v1",
        baseline_model_id="baseline-model",
        baseline_model_revision="baseline-model-revision",
    )
    registry.register_candidate(valid_candidate_manifest("candidate-v2"))

    promoted = registry.promote(
        "candidate-v2",
        report=eligible_report(),
        approval=approval(),
    )

    assert promoted.active_model == "candidate-v2"
    assert registry.read().previous_model == "baseline-v1"


def test_concurrent_candidate_registrations_preserve_both_during_race(tmp_path, monkeypatch):
    import threading

    registry = DistillModelRegistry(tmp_path / "registry.json")
    barrier = threading.Barrier(2)
    original = DistillModelRegistry.register_candidate

    def synchronized_register(self, manifest):
        barrier.wait()
        return original(self, manifest)

    monkeypatch.setattr(DistillModelRegistry, "register_candidate", synchronized_register)

    threads = [
        threading.Thread(target=lambda: registry.register_candidate(valid_candidate_manifest("candidate-v2"))),
        threading.Thread(target=lambda: registry.register_candidate(valid_candidate_manifest("candidate-v3"))),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    stored = registry.read()
    assert set(stored.candidates) == {"candidate-v2", "candidate-v3"}


def test_promotion_requires_benchmark_eligibility_and_approval(tmp_path):
    registry = setup_registry(tmp_path)
    report = eligible_report()
    incomplete_report = report.model_copy(
        update={
            "metadata": report.metadata.model_copy(update={"eval_set_version": ""})
        }
    )

    with pytest.raises(PromotionError, match="eligible"):
        registry.promote(
            "candidate-v2",
            report=ineligible_benchmark_report(),
            approval=approval(),
        )
    with pytest.raises(PromotionError, match="eligible"):
        registry.promote("candidate-v2", report=incomplete_report, approval=approval())
    with pytest.raises(PromotionError, match="approval"):
        registry.promote("candidate-v2", report=eligible_report(), approval=None)
    invalid_approval = approval().model_copy(
        update={"confirmed_at": now_utc().replace(tzinfo=None)}
    )
    with pytest.raises(PromotionError, match="invalid"):
        registry.promote(
            "candidate-v2",
            report=eligible_report(),
            approval=invalid_approval,
        )

    assert registry.read().active_model == "baseline-v1"
    assert registry.read().previous_model is None


def test_approval_requires_operator_and_timezone_aware_confirmation():
    with pytest.raises(ValidationError):
        HumanApproval(approved_by=" ", confirmed_at=now_utc())
    with pytest.raises(ValidationError):
        HumanApproval(
            approved_by="operator",
            confirmed_at=now_utc().replace(tzinfo=None),
        )


def test_valid_promotion_preserves_previous_and_rolls_back_without_deleting_tags(tmp_path):
    registry = setup_registry(tmp_path)

    promoted = registry.promote(
        "candidate-v2",
        report=eligible_report(),
        approval=approval(),
    )

    assert promoted.active_model == "candidate-v2"
    assert promoted.previous_model == "baseline-v1"
    assert set(promoted.candidates) == {"candidate-v2"}
    assert registry.resolve_active_model_name("tuminh-active") == "candidate-v2"

    rolled_back = registry.rollback()

    assert rolled_back.active_model == "baseline-v1"
    assert rolled_back.previous_model == "candidate-v2"
    assert set(rolled_back.candidates) == {"candidate-v2"}
    assert registry.resolve_active_model_name("tuminh-active") == "baseline-v1"


def test_initialization_is_explicit_and_only_when_no_active_model(tmp_path):
    registry = DistillModelRegistry(tmp_path / "registry.json")

    assert registry.read().active_model is None
    registry.initialize_active("baseline-v1")

    with pytest.raises(RegistryError, match="already"):
        registry.initialize_active("baseline-v2")
    with pytest.raises(RegistryError):
        DistillModelRegistry(tmp_path / "uninitialized.json").rollback()


@pytest.mark.parametrize(
    "report",
    [
        eligible_report().model_copy(
            update={
                "metadata": eligible_report().metadata.model_copy(
                    update={"candidate_model_revision": "other-revision"}
                )
            }
        ),
        eligible_report().model_copy(
            update={
                "metadata": eligible_report().metadata.model_copy(
                    update={"dataset_hash": "f" * 64}
                )
            }
        ),
        eligible_report().model_copy(
            update={
                "metadata": eligible_report().metadata.model_copy(
                    update={"benchmark_id": "other-benchmark"}
                )
            }
        ),
    ],
)
def test_promotion_rejects_report_with_mismatched_candidate_metadata(tmp_path, report):
    registry = setup_registry(tmp_path)

    with pytest.raises(PromotionError):
        registry.promote("candidate-v2", report=report, approval=approval())

    assert registry.read().active_model == "baseline-v1"


def test_promotion_rejects_report_for_a_different_candidate_model(tmp_path):
    registry = setup_registry(tmp_path)
    report = eligible_report()
    other_model = "different-model"
    mismatched_report = report.model_copy(
        update={
            "metadata": report.metadata.model_copy(
                update={"candidate_model_id": other_model}
            ),
            "candidate_score": report.candidate_score.model_copy(
                update={"model_id": other_model}
            ),
        }
    )
    assert mismatched_report.eligible_for_promotion is True

    with pytest.raises(PromotionError, match="metadata"):
        registry.promote(
            "candidate-v2", report=mismatched_report, approval=approval()
        )

    assert registry.read().active_model == "baseline-v1"


def test_promotion_rejects_unknown_candidate_and_invalid_candidate_ids(tmp_path):
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active("baseline-v1")

    invalid_manifest = valid_candidate_manifest("candidate-v2").model_copy(
        update={"candidate_id": "../escape"}
    )
    with pytest.raises(RegistryError):
        registry.register_candidate(invalid_manifest)
    with pytest.raises(PromotionError, match="registered"):
        registry.promote("unknown", report=eligible_report(), approval=approval())
    assert registry.read().active_model == "baseline-v1"


def test_candidate_registration_rejects_initialized_baseline_tag_without_changing_registry(
    tmp_path,
):
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active(
        "baseline-v1",
        baseline_model_id="baseline-model",
        baseline_model_revision="baseline-model-revision",
    )
    original_bytes = registry.path.read_bytes()
    original_state = registry.read()

    with pytest.raises(RegistryError, match="baseline"):
        registry.register_candidate(valid_candidate_manifest("baseline-v1"))

    assert registry.path.read_bytes() == original_bytes
    assert registry.read() == original_state


def test_registered_candidate_is_not_promoted_until_explicit_approval(tmp_path):
    registry = setup_registry(tmp_path)

    assert registry.read().active_model == "baseline-v1"
    assert registry.resolve_active_model_name("tuminh-active") == "baseline-v1"


def test_invalid_registry_json_never_gets_replaced_by_a_write(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text('{"active_model": "../invalid"}', encoding="utf-8")
    original = path.read_bytes()
    registry = DistillModelRegistry(path)

    with pytest.raises(RegistryError):
        registry.register_candidate(valid_candidate_manifest("candidate-v2"))

    assert path.read_bytes() == original


def test_registry_document_requires_supported_schema_version(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text('{"active_model": null}', encoding="utf-8")
    original = path.read_bytes()
    registry = DistillModelRegistry(path)

    with pytest.raises(RegistryError):
        registry.register_candidate(valid_candidate_manifest("candidate-v2"))

    assert path.read_bytes() == original


def test_failed_atomic_replacement_keeps_last_valid_registry(tmp_path, monkeypatch):
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active("baseline-v1")
    original = registry.path.read_bytes()

    def interrupted_replace(source, destination):
        raise OSError("simulated interruption")

    monkeypatch.setattr(registry_module.os, "replace", interrupted_replace)

    with pytest.raises(RegistryError, match="write"):
        registry.register_candidate(valid_candidate_manifest("candidate-v2"))

    assert registry.path.read_bytes() == original
    assert DistillModelRegistry(registry.path).read().active_model == "baseline-v1"
    assert list(tmp_path.glob("registry.json.*.tmp")) == []


def test_active_resolution_rejects_unknown_alias_and_corrupt_installed_reference(tmp_path):
    registry = setup_registry(tmp_path)

    with pytest.raises(RegistryError):
        registry.resolve_active_model_name("other-alias")

    payload = json.loads(registry.path.read_text(encoding="utf-8"))
    payload["active_model"] = "candidate-v2"
    del payload["candidates"]["candidate-v2"]
    registry.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RegistryError):
        registry.resolve_active_model_name("tuminh-active")
