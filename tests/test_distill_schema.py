import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from distill.schema import (
    CandidateManifest,
    DatasetManifest,
    ExternalExportHeader,
    ImportProvenance,
    ObservableEvent,
    TrainingMessage,
    TrainingRow,
    TrajectoryExport,
    parse_trajectory_jsonl,
)
from tests.distill_fakes import now_utc, valid_trajectory


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
def test_training_message_accepts_observable_roles(role):
    message = TrainingMessage(role=role, content="Synthetic fixture content.")

    assert message.role == role


@pytest.mark.parametrize("role", ["thought", "system", "developer", "analysis"])
def test_training_message_rejects_private_or_prompt_roles(role):
    with pytest.raises(ValidationError):
        TrainingMessage(role=role, content="Synthetic fixture content.")


def test_training_row_requires_an_observable_message_and_provenance():
    row = TrainingRow(
        task_id="task-1",
        messages=[
            TrainingMessage(role="user", content="Do a safe task."),
            TrainingMessage(role="assistant", content="Task complete."),
        ],
        verification_passed=True,
        eval_passed=True,
        provenance_id="source-1",
    )

    assert row.provenance_id == "source-1"
    with pytest.raises(ValidationError):
        TrainingRow(
            task_id="task-1",
            messages=[],
            verification_passed=True,
            eval_passed=True,
            provenance_id="source-1",
        )


def test_external_trajectory_requires_authorization_metadata():
    record = valid_trajectory()
    payload = record.model_dump(mode="json")
    payload["source_type"] = "external"
    payload["provenance"]["license_or_authorization"] = None
    payload["provenance"]["authorized_by"] = None

    with pytest.raises(ValidationError, match="authorization"):
        TrajectoryExport.model_validate(payload)


def test_invalid_task_outcome_is_rejected():
    payload = valid_trajectory().model_dump(mode="json")
    payload["task_outcome"] = "incomplete"

    with pytest.raises(ValidationError):
        TrajectoryExport.model_validate(payload)


def test_duplicate_event_sequence_numbers_are_rejected():
    payload = valid_trajectory().model_dump(mode="json")
    payload["events"][1]["sequence_number"] = payload["events"][0]["sequence_number"]

    with pytest.raises(ValidationError, match="sequence"):
        TrajectoryExport.model_validate(payload)


@pytest.mark.parametrize(
    "private_key",
    [
        "thought",
        "thinking",
        "analysis",
        "system_prompt",
        "developer_prompt",
        "credential",
        "environment",
        "API_KEY",
    ],
)
def test_nested_private_fields_are_rejected(private_key):
    payload = valid_trajectory().model_dump(mode="json")
    payload["events"][0]["data"] = {"nested": [{private_key: "not for training"}]}

    with pytest.raises(ValidationError, match="prohibited"):
        TrajectoryExport.model_validate(payload)


def test_valid_runtime_jsonl_fixture_parses_as_one_trajectory():
    fixture = Path(__file__).parent / "fixtures" / "distill" / "valid-runtime-export.jsonl"

    trajectory = parse_trajectory_jsonl(fixture.read_text(encoding="utf-8"))

    assert trajectory.task_id == "task-runtime-1"
    assert trajectory.source_type == "runtime"
    assert trajectory.task_outcome == "succeeded"
    assert [event.event_type for event in trajectory.events] == ["task", "verification"]


def test_authorized_external_fixture_has_no_private_roles():
    fixture = Path(__file__).parent / "fixtures" / "distill" / "authorized-external-export.jsonl"
    lines = fixture.read_text(encoding="utf-8").splitlines()

    messages = [
        TrainingMessage.model_validate(ObservableEvent.model_validate_json(line).data["message"])
        for line in lines[1:]
    ]
    assert [message.role for message in messages] == ["user", "assistant", "tool"]


def test_external_fixture_with_nested_private_field_is_rejected():
    fixture = Path(__file__).parent / "fixtures" / "distill" / "invalid-private-fields.jsonl"

    with pytest.raises(ValidationError, match="prohibited"):
        ObservableEvent.model_validate_json(fixture.read_text(encoding="utf-8").splitlines()[1])


def test_dataset_manifest_validates_hashes_and_counts():
    manifest = DatasetManifest(
        dataset_version="v1",
        source_hashes=["a" * 64],
        curation_version="1.0",
        redaction_counts={"secrets": 2},
        created_at=now_utc(),
        row_count=1,
    )

    assert manifest.row_count == 1
    with pytest.raises(ValidationError):
        DatasetManifest(
            dataset_version="v1",
            source_hashes=["not-a-hash"],
            curation_version="1.0",
            redaction_counts={"secrets": -1},
            created_at=now_utc(),
            row_count=-1,
        )


def test_candidate_manifest_requires_model_license_and_dataset_hash():
    candidate = CandidateManifest(
        candidate_id="candidate-v1",
        model_id="org/model",
        model_revision="revision-1",
        base_model_license="declared license",
        dataset_hash="b" * 64,
        training_config={"rank": 8},
        runtime_metadata={"python": "3.11"},
        hardware_metadata={"device": "cpu"},
        benchmark_id="benchmark-1",
    )

    assert candidate.candidate_id == "candidate-v1"
    invalid = candidate.model_dump()
    invalid["dataset_hash"] = "invalid"
    with pytest.raises(ValidationError):
        CandidateManifest.model_validate(invalid)


def test_runtime_header_fixture_is_valid_json():
    fixture = Path(__file__).parent / "fixtures" / "distill" / "valid-runtime-export.jsonl"
    header = json.loads(fixture.read_text(encoding="utf-8").splitlines()[0])

    assert header["record_type"] == "header"
    assert header["schema_version"] == 1


def test_external_export_header_does_not_supply_authorization_by_itself():
    fixture = Path(__file__).parent / "fixtures" / "distill" / "authorized-external-export.jsonl"
    header = ExternalExportHeader.model_validate_json(
        fixture.read_text(encoding="utf-8").splitlines()[0]
    )

    assert header.task_id == "task-external-1"


def test_import_provenance_accepts_optional_source_model_identity():
    provenance = ImportProvenance(
        source_name="claude",
        license_or_authorization="Synthetic test authorization.",
        authorized_by="test-operator",
        imported_at=now_utc(),
        source_sha256="a" * 64,
        source_model_id="provider/model",
        source_model_revision="revision-1",
    )

    assert provenance.source_model_id == "provider/model"
    assert provenance.source_model_revision == "revision-1"


def test_import_provenance_defaults_source_model_identity_for_legacy_records():
    provenance = ImportProvenance(
        source_name="legacy-agent",
        license_or_authorization="Synthetic test authorization.",
        authorized_by="test-operator",
        imported_at=now_utc(),
        source_sha256="a" * 64,
    )

    assert provenance.source_model_id is None
    assert provenance.source_model_revision is None


def test_import_provenance_rejects_revision_without_model_id():
    with pytest.raises(ValidationError, match="source_model_id"):
        ImportProvenance(
            source_name="claude",
            license_or_authorization="Synthetic test authorization.",
            authorized_by="test-operator",
            imported_at=now_utc(),
            source_sha256="a" * 64,
            source_model_revision="revision-1",
        )
