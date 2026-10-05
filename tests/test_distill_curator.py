import json
import hashlib
from pathlib import Path

import pytest

from distill.curator import Curator, CuratorError
from distill.schema import ImportProvenance, TrajectoryExport, TrainingRow
from tests.distill_fakes import now_utc, valid_trajectory


def record_with_source_hash(source_hash: str, **changes) -> TrajectoryExport:
    record = valid_trajectory()
    payload = record.model_dump(mode="python")
    payload["provenance"] = {
        **record.provenance.model_dump(mode="python"),
        "source_sha256": source_hash,
    }
    payload.update(changes)
    return TrajectoryExport.model_validate(payload)


def test_curator_excludes_failed_or_unverified_trajectories(tmp_path):
    failed = record_with_source_hash(
        "a" * 64,
        task_outcome="failed",
        verification_passed=False,
        eval_passed=False,
    )
    unverified = record_with_source_hash(
        "b" * 64,
        task_outcome="failed",
        verification_passed=False,
        eval_passed=True,
    )
    output = tmp_path / "curated" / "dataset.jsonl"

    manifest = Curator(secret_values=[]).curate([failed, unverified], output)

    assert manifest.row_count == 0
    assert output.read_text(encoding="utf-8") == ""
    assert manifest.source_hashes == []
    assert output.with_suffix(".manifest.json").exists()


def test_curator_redacts_configured_secrets_in_training_messages(tmp_path):
    record = valid_trajectory()
    events = [
        event.model_copy(
            update={
                "data": {
                    "message": {
                        "role": "assistant",
                        "content": "The private-marker is not for training.",
                    }
                },
                "event_type": "assistant_message",
                "sequence_number": 2,
            }
        )
        if event.sequence_number == 2
        else event
        for event in record.events
    ]
    events.append(
        record.events[-1].model_copy(
            update={
                "event_type": "verification",
                "sequence_number": 3,
                "data": {"passed": True},
            }
        )
    )
    record = TrajectoryExport.model_validate(
        {
            **record.model_dump(mode="python"),
            "events": [event.model_dump(mode="python") for event in events],
        }
    )
    output = tmp_path / "dataset.jsonl"

    manifest = Curator(secret_values=["private-marker"]).curate([record], output)

    row = TrainingRow.model_validate_json(output.read_text(encoding="utf-8").strip())
    serialized = output.read_text(encoding="utf-8")
    assert "private-marker" not in serialized
    assert "[REDACTED]" in serialized
    assert manifest.redaction_counts["configured_secret_values"] == 1
    assert any(message.role == "assistant" for message in row.messages)


def test_manifest_records_source_hash_and_authorization_provenance(tmp_path):
    record = valid_trajectory()
    source_hash = "c" * 64
    provenance = ImportProvenance(
        source_name="authorized-export",
        license_or_authorization="Operator confirms usage rights.",
        authorized_by="operator",
        imported_at=now_utc(),
        source_sha256=source_hash,
    )
    record = TrajectoryExport.model_validate(
        {
            **record.model_dump(mode="python"),
            "source_type": "external",
            "provenance": provenance.model_dump(mode="python"),
        }
    )

    manifest = Curator(secret_values=[]).curate([record], tmp_path / "dataset.jsonl")

    assert manifest.source_hashes == [source_hash]
    assert manifest.source_provenance[0].license_or_authorization == (
        "Operator confirms usage rights."
    )
    assert manifest.source_provenance[0].authorized_by == "operator"


def test_invalid_batch_preserves_previous_dataset_and_manifest(tmp_path):
    output = tmp_path / "dataset.jsonl"
    manifest_path = output.with_suffix(".manifest.json")
    output.write_text("previous data\n", encoding="utf-8")
    manifest_path.write_text("previous manifest\n", encoding="utf-8")

    with pytest.raises(CuratorError):
        Curator(secret_values=[]).curate([{"invalid": "record"}], output)

    assert output.read_text(encoding="utf-8") == "previous data\n"
    assert manifest_path.read_text(encoding="utf-8") == "previous manifest\n"


def test_curator_purge_removes_only_matching_source_rows_and_raw_file(tmp_path):
    raw_one = b"authorized synthetic raw source one"
    raw_two = b"authorized synthetic raw source two"
    source_one = hashlib.sha256(raw_one).hexdigest()
    source_two = hashlib.sha256(raw_two).hexdigest()
    record_one = record_with_source_hash(source_one)
    record_two = record_with_source_hash(source_two)
    raw_root = tmp_path / "raw"
    curated_root = tmp_path / "curated"
    raw_root.mkdir()
    (raw_root / f"{source_one}.jsonl").write_bytes(raw_one)
    (raw_root / f"{source_two}.jsonl").write_bytes(raw_two)
    dataset = curated_root / "dataset.jsonl"
    Curator(secret_values=[]).curate([record_one, record_two], dataset)

    result = Curator(secret_values=[]).purge(
        source_one,
        raw_root=raw_root,
        curated_root=curated_root,
    )

    rows = [
        TrainingRow.model_validate_json(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(dataset.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert result.raw_files_removed == 1
    assert result.curated_rows_removed == 1
    assert [row.provenance_id for row in rows] == [source_two]
    assert manifest["source_hashes"] == [source_two]
    assert not (raw_root / f"{source_one}.jsonl").exists()
    assert (raw_root / f"{source_two}.jsonl").exists()


def test_purge_rejects_invalid_source_hash_and_does_not_remove_data(tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = raw_root / "source.jsonl"
    raw.write_text("keep", encoding="utf-8")

    with pytest.raises(CuratorError, match="SHA-256"):
        Curator(secret_values=[]).purge(
            "../source",
            raw_root=raw_root,
            curated_root=tmp_path / "curated",
        )

    assert raw.read_text(encoding="utf-8") == "keep"


def test_purge_validates_all_datasets_before_modifying_any(tmp_path):
    raw = b"authorized raw data"
    source_hash = hashlib.sha256(raw).hexdigest()
    raw_root = tmp_path / "raw"
    curated_root = tmp_path / "curated"
    raw_root.mkdir()
    curated_root.mkdir()
    (raw_root / f"{source_hash}.jsonl").write_bytes(raw)
    first_dataset = curated_root / "a-valid.jsonl"
    Curator(secret_values=[]).curate(
        [record_with_source_hash(source_hash)], first_dataset
    )
    first_manifest = first_dataset.with_suffix(".manifest.json")
    before = {
        first_dataset: first_dataset.read_bytes(),
        first_manifest: first_manifest.read_bytes(),
    }
    invalid_dataset = curated_root / "z-invalid.jsonl"
    invalid_dataset.write_text("not a valid training row\n", encoding="utf-8")
    invalid_manifest = invalid_dataset.with_suffix(".manifest.json")
    invalid_manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(CuratorError, match="invalid"):
        Curator(secret_values=[]).purge(source_hash, raw_root, curated_root)

    assert {path: path.read_bytes() for path in before} == before
    assert (raw_root / f"{source_hash}.jsonl").read_bytes() == raw


def test_purge_refuses_symlinked_manifest(tmp_path):
    raw = b"authorized raw data"
    source_hash = hashlib.sha256(raw).hexdigest()
    raw_root = tmp_path / "raw"
    curated_root = tmp_path / "curated"
    raw_root.mkdir()
    curated_root.mkdir()
    (raw_root / f"{source_hash}.jsonl").write_bytes(raw)
    dataset = curated_root / "dataset.jsonl"
    Curator(secret_values=[]).curate(
        [record_with_source_hash(source_hash)], dataset
    )
    manifest_path = dataset.with_suffix(".manifest.json")
    outside_manifest = tmp_path / "outside.manifest.json"
    outside_manifest.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    try:
        manifest_path.symlink_to(outside_manifest)
    except OSError as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")
    before_dataset = dataset.read_bytes()

    with pytest.raises(CuratorError, match="symlink"):
        Curator(secret_values=[]).purge(source_hash, raw_root, curated_root)

    assert dataset.read_bytes() == before_dataset
    assert manifest_path.is_symlink()
    assert outside_manifest.exists()
    assert (raw_root / f"{source_hash}.jsonl").read_bytes() == raw
