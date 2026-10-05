from datetime import UTC
import hashlib
from pathlib import Path

import pytest
from agent.trajectory import TrajectoryJournal
from distill.importers import (
    DistillImportError,
    ExternalImporter,
    RuntimeExporter,
)
from distill.cli import main as distill_main
from distill.schema import parse_trajectory_jsonl


FIXTURES = Path(__file__).parent / "fixtures" / "distill"


def make_runtime_journal(root: Path) -> TrajectoryJournal:
    journal = TrajectoryJournal(root / "memory" / "trajectories", ["synthetic-secret"])
    journal.append("task-runtime-7", "task", {"title": "Inspect safe behavior"})
    journal.append("task-runtime-7", "verification", {"passed": True})
    journal.append("task-runtime-7", "reflection", {"outcome": "succeeded"})
    return journal


def test_runtime_export_writes_one_validated_local_trajectory(tmp_path):
    journal = make_runtime_journal(tmp_path)
    output = tmp_path / "exports" / "task-7.jsonl"

    exported = RuntimeExporter(journal.root).export("task-runtime-7", output)

    trajectory = parse_trajectory_jsonl(exported.read_text(encoding="utf-8"))
    assert exported == output
    assert trajectory.task_id == "task-runtime-7"
    assert trajectory.source_type == "runtime"
    assert trajectory.task_outcome == "succeeded"
    assert trajectory.verification_passed is True
    assert trajectory.eval_passed is None
    assert len(trajectory.events) == 3


def test_runtime_export_rejects_traversal_task_ids(tmp_path):
    exporter = RuntimeExporter(tmp_path / "trajectories")

    with pytest.raises(DistillImportError, match="task"):
        exporter.export("../outside", tmp_path / "output.jsonl")


def test_runtime_export_does_not_replace_output_if_input_is_invalid(tmp_path):
    journal_root = tmp_path / "trajectories"
    journal_root.mkdir()
    (journal_root / "task-invalid.jsonl").write_text("{broken\n", encoding="utf-8")
    output = tmp_path / "output.jsonl"
    output.write_text("previous export\n", encoding="utf-8")

    with pytest.raises(DistillImportError):
        RuntimeExporter(journal_root).export("task-invalid", output)

    assert output.read_text(encoding="utf-8") == "previous export\n"


def test_external_import_requires_operator_authorization(tmp_path):
    source = tmp_path / "external.jsonl"
    source.write_text(
        (FIXTURES / "authorized-external-export.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(DistillImportError, match="authorization"):
        ExternalImporter().import_file(
            source,
            source_name="synthetic-provider-export",
            license_or_authorization=" ",
            authorized_by="operator",
        )


def test_external_import_records_provenance_and_source_hash(tmp_path):
    source = tmp_path / "authorized.jsonl"
    source.write_bytes((FIXTURES / "authorized-external-export.jsonl").read_bytes())
    source_bytes = source.read_bytes()

    imported = ExternalImporter().import_file(
        source,
        source_name="synthetic-provider-export",
        license_or_authorization="Synthetic fixture authorization.",
        authorized_by="test-operator",
    )

    assert len(imported) == 1
    trajectory = imported[0]
    assert trajectory.source_type == "external"
    assert trajectory.provenance.source_name == "synthetic-provider-export"
    assert trajectory.provenance.license_or_authorization == "Synthetic fixture authorization."
    assert trajectory.provenance.authorized_by == "test-operator"
    assert trajectory.provenance.source_sha256 == hashlib.sha256(source_bytes).hexdigest()
    assert trajectory.provenance.imported_at.tzinfo == UTC


def test_external_import_accepts_multiple_unique_trajectories(tmp_path):
    source = tmp_path / "multiple.jsonl"
    fixture = FIXTURES / "authorized-external-export.jsonl"
    first_block = fixture.read_text(encoding="utf-8")
    second_block = (
        first_block.replace(
            '"trajectory-external-1"',
            '"trajectory-external-2"',
        )
        .replace('"task-external-1"', '"task-external-2"')
    )
    source.write_text(first_block + second_block, encoding="utf-8")

    imported = ExternalImporter().import_file(
        source,
        source_name="deepseek",
        license_or_authorization="Synthetic fixture authorization.",
        authorized_by="test-operator",
    )

    assert [record.trajectory_id for record in imported] == [
        "trajectory-external-1",
        "trajectory-external-2",
    ]
    assert imported[0].provenance.source_sha256 == imported[1].provenance.source_sha256


def test_external_import_records_source_model_identity(tmp_path):
    source = tmp_path / "agent.jsonl"
    source.write_bytes((FIXTURES / "authorized-external-export.jsonl").read_bytes())

    imported = ExternalImporter().import_file(
        source,
        source_name="claude",
        license_or_authorization="Synthetic fixture authorization.",
        authorized_by="test-operator",
        source_model_id="provider/model",
        source_model_revision="revision-1",
    )

    assert imported[0].provenance.source_model_id == "provider/model"
    assert imported[0].provenance.source_model_revision == "revision-1"


def test_external_import_rejects_revision_without_model_id(tmp_path):
    source = tmp_path / "agent.jsonl"
    source.write_bytes((FIXTURES / "authorized-external-export.jsonl").read_bytes())

    with pytest.raises(DistillImportError, match="provenance"):
        ExternalImporter().import_file(
            source,
            source_name="claude",
            license_or_authorization="Synthetic fixture authorization.",
            authorized_by="test-operator",
            source_model_revision="revision-1",
        )


def test_external_import_rejects_source_over_size_limit(tmp_path, monkeypatch):
    source = tmp_path / "oversized.jsonl"
    source.write_bytes((FIXTURES / "authorized-external-export.jsonl").read_bytes())
    monkeypatch.setattr(
        ExternalImporter,
        "_MAX_SOURCE_FILE_BYTES",
        source.stat().st_size - 1,
    )

    with pytest.raises(DistillImportError, match="size limit"):
        ExternalImporter().import_file(
            source,
            source_name="claude",
            license_or_authorization="Synthetic fixture authorization.",
            authorized_by="test-operator",
        )


def test_external_import_rejects_event_over_encoded_size_limit(tmp_path, monkeypatch):
    source = tmp_path / "large-event.jsonl"
    source.write_bytes((FIXTURES / "authorized-external-export.jsonl").read_bytes())
    monkeypatch.setattr(ExternalImporter, "_MAX_EVENT_BYTES", 64)

    with pytest.raises(DistillImportError, match="event.*size limit"):
        ExternalImporter().import_file(
            source,
            source_name="claude",
            license_or_authorization="Synthetic fixture authorization.",
            authorized_by="test-operator",
        )


def test_external_import_rejects_too_many_events_per_trajectory(tmp_path, monkeypatch):
    source = tmp_path / "too-many-events.jsonl"
    source.write_bytes((FIXTURES / "authorized-external-export.jsonl").read_bytes())
    monkeypatch.setattr(ExternalImporter, "_MAX_EVENTS_PER_TRAJECTORY", 2)

    with pytest.raises(DistillImportError, match="event count limit"):
        ExternalImporter().import_file(
            source,
            source_name="claude",
            license_or_authorization="Synthetic fixture authorization.",
            authorized_by="test-operator",
        )


def test_external_import_rejects_duplicate_trajectory_ids_within_file(tmp_path):
    source = tmp_path / "duplicate.jsonl"
    block = (FIXTURES / "authorized-external-export.jsonl").read_text(encoding="utf-8")
    source.write_text(block + block, encoding="utf-8")

    with pytest.raises(DistillImportError, match="duplicate trajectory id"):
        ExternalImporter().import_file(
            source,
            source_name="claude",
            license_or_authorization="Synthetic fixture authorization.",
            authorized_by="test-operator",
        )


def test_external_import_rejects_private_reasoning_without_partial_write(tmp_path):
    source = tmp_path / "invalid.jsonl"
    source.write_bytes((FIXTURES / "invalid-private-fields.jsonl").read_bytes())
    raw_dir = tmp_path / "distill" / "data" / "raw"

    with pytest.raises(DistillImportError):
        ExternalImporter().import_file(
            source,
            source_name="synthetic-provider-export",
            license_or_authorization="Synthetic fixture authorization.",
            authorized_by="test-operator",
        )

    assert not raw_dir.exists()


def test_external_import_rejects_private_message_roles(tmp_path):
    source = tmp_path / "system-role.jsonl"
    lines = (FIXTURES / "authorized-external-export.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    lines[1] = lines[1].replace('"role":"user"', '"role":"system"')
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(DistillImportError, match="role"):
        ExternalImporter().import_file(
            source,
            source_name="synthetic-provider-export",
            license_or_authorization="Synthetic fixture authorization.",
            authorized_by="test-operator",
        )


def test_external_import_rejects_credential_like_message_content(tmp_path):
    source = tmp_path / "credential.jsonl"
    lines = (FIXTURES / "authorized-external-export.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    lines[1] = lines[1].replace(
        "Add a safe check.",
        "Use Bearer " + "synthetic-token-value-123456",
    )
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(DistillImportError, match="credential"):
        ExternalImporter().import_file(
            source,
            source_name="synthetic-provider-export",
            license_or_authorization="Synthetic fixture authorization.",
            authorized_by="test-operator",
        )


def test_external_import_reports_malformed_jsonl(tmp_path):
    source = tmp_path / "malformed.jsonl"
    source.write_text("{broken\n", encoding="utf-8")

    with pytest.raises(DistillImportError, match="malformed"):
        ExternalImporter().import_file(
            source,
            source_name="synthetic-provider-export",
            license_or_authorization="Synthetic fixture authorization.",
            authorized_by="test-operator",
        )


def test_distillation_cli_exports_runtime_trajectory(tmp_path):
    journal = make_runtime_journal(tmp_path)
    output = tmp_path / "exports" / "runtime.jsonl"

    result = distill_main(
        [
            "export-runtime",
            "--journal-root",
            str(journal.root),
            "--task-id",
            "task-runtime-7",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert parse_trajectory_jsonl(output.read_text(encoding="utf-8")).task_id == "task-runtime-7"


def test_distillation_cli_imports_authorized_external_export(tmp_path):
    source = tmp_path / "external.jsonl"
    source.write_bytes((FIXTURES / "authorized-external-export.jsonl").read_bytes())
    output = tmp_path / "raw" / "external.jsonl"

    result = distill_main(
        [
            "import-external",
            str(source),
            "--source-name",
            "synthetic-provider-export",
            "--license-or-authorization",
            "Synthetic fixture authorization.",
            "--authorized-by",
            "test-operator",
            "--output",
            str(output),
        ]
    )

    imported = parse_trajectory_jsonl(output.read_text(encoding="utf-8"))
    assert result == 0
    assert imported.source_type == "external"
    assert imported.provenance.authorized_by == "test-operator"
