# Multi-Agent Distillation (TADP/1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Formalize and harden the existing local JSONL import path so authorized observable trajectories from Claude, Antigravity, DeepSeek, and other agents can enter Tự Minh's current curation pipeline.

**Architecture:** Keep the existing version-1 JSONL header/event format and `import-external` command as the provider-neutral TADP/1 boundary. Extend provenance with optional source model identity, apply whole-file size/event-count limits before accepting imports, and keep the existing curator as the only gate to training rows.

**Tech Stack:** Python 3.11+, Pydantic 2, pytest 8, argparse, JSONL.

**Spec:** `docs/superpowers/specs/2026-10-05-multi-agent-distillation-design.md`

## Global Constraints

- Keep TADP/1 on `schema_version: 1` and preserve compatibility with existing external fixtures.
- Maximum source file size: 64 MiB.
- Maximum events per trajectory: 1,000.
- Maximum encoded event size: 64 KiB.
- Require operator-provided source name, license/authorization declaration, and authorizing operator for every external import.
- Reject hidden/private fields and credential-like values before writing; errors must not include rejected content.
- Validate the whole file before writing any output; importing must not execute content, scrape sessions, upload data, or initiate training.
- Keep curation eligibility unchanged: successful outcome, passing verification, and passing evaluation are required.
- Do not add dependencies, provider API clients, provider-specific parsers, or training/deployment behavior.

## File Structure

- `distill/schema.py` — add optional source-model identity to strict provenance and enforce that a revision is not supplied without a model ID.
- `distill/importers.py` — accept source-model provenance and enforce the three TADP/1 size/count limits while validating all trajectory blocks.
- `distill/cli.py` — expose optional model-provenance flags on `import-external` and pass them through without changing its authorization requirements.
- `tests/test_distill_schema.py` — cover optional provenance, legacy defaults, and invalid model/revision combinations.
- `tests/test_distill_importers.py` — cover metadata propagation, multiple blocks, duplicate IDs, and input limits.
- `tests/test_distill_cli.py` — cover CLI argument help, metadata persistence, and safe failure with no partial output or leaked message content.
- `README.md` — document the TADP/1 event contract, explicit local import command, provenance flags, and adapter boundary.

---

### Task 1: Extending provenance with optional model identity

**Files:**
- Modify: `distill/schema.py`
- Test: `tests/test_distill_schema.py`

**Interfaces:**
- `ImportProvenance` gains `source_model_id: str | None = None` and `source_model_revision: str | None = None`.
- A non-null `source_model_revision` requires a non-null `source_model_id`; existing records with neither field remain valid.

- [ ] **Step 1: Add failing provenance tests**

Append these tests to `tests/test_distill_schema.py`:

```python
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
```

- [ ] **Step 2: Run the focused tests and confirm the new fields fail**

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_distill_schema.py -q
```

Expected: the three new tests fail because provenance does not yet accept or validate source-model fields; existing schema tests remain unchanged.

- [ ] **Step 3: Implement the provenance fields and invariant**

In `ImportProvenance`, add the two optional string fields with non-empty validation when supplied and a separate after-validator:

```python
    source_model_id: str | None = Field(default=None, min_length=1)
    source_model_revision: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_source_model_for_revision(self) -> ImportProvenance:
        if self.source_model_revision is not None and self.source_model_id is None:
            raise ValueError("source_model_revision requires source_model_id")
        return self
```

Keep both defaults as `None` so existing external, runtime, and dataset-manifest provenance continues to validate.

- [ ] **Step 4: Run schema and existing manifest compatibility tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_distill_schema.py -q
```

Expected: all schema tests pass, including the legacy provenance and existing version-1 fixture checks.

- [ ] **Step 5: Commit**

```powershell
git add distill/schema.py tests/test_distill_schema.py
git commit -m "feat: extend trajectory source provenance"
```

### Task 2: Enforcing TADP/1 import limits and provenance

**Files:**
- Modify: `distill/importers.py`
- Test: `tests/test_distill_importers.py`

**Interfaces:**
- `ExternalImporter.import_file(path, *, source_name, license_or_authorization, authorized_by, source_model_id=None, source_model_revision=None)` returns validated `list[TrajectoryExport]`.
- `DistillImportError` remains the public import failure type.
- `_split_export_blocks` receives explicit event-count and encoded-event-byte limits from `ExternalImporter`; it continues to reject malformed framing and duplicate trajectory IDs before returning blocks.

- [ ] **Step 1: Add failing tests for source-model metadata and TADP limits**

Add tests to `tests/test_distill_importers.py`. Use the existing `FIXTURES / "authorized-external-export.jsonl"` fixture and authorized metadata on every valid import.

```python
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
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_distill_importers.py -q
```

Expected: the multiple-trajectory regression passes on the existing parser; the new provenance keyword, revision guard, and limit assertions fail; existing import behavior tests remain green.

- [ ] **Step 3: Implement the importer limits**

Add these class constants to `ExternalImporter`:

```python
_MAX_SOURCE_FILE_BYTES = 64 * 1024 * 1024
_MAX_EVENTS_PER_TRAJECTORY = 1_000
_MAX_EVENT_BYTES = 64 * 1024
```

In `import_file`, check the size before and after reading and convert provenance validation failures to a content-free `DistillImportError`:

```python
        source_path = Path(path)
        try:
            if source_path.stat().st_size > self._MAX_SOURCE_FILE_BYTES:
                raise DistillImportError(
                    f"{source_path}: External export exceeds the source size limit"
                )
            source_bytes = source_path.read_bytes()
            content = source_bytes.decode("utf-8")
        except DistillImportError:
            raise
        except (OSError, UnicodeError) as error:
            raise DistillImportError(
                f"{source_path}: Could not read the supplied external export"
            ) from error
        if len(source_bytes) > self._MAX_SOURCE_FILE_BYTES:
            raise DistillImportError(
                f"{source_path}: External export exceeds the source size limit"
            )
```

Pass `source_model_id` and `source_model_revision` into `ImportProvenance`. Wrap its `ValidationError` with `DistillImportError("External import provenance is invalid")` so callers receive the public import error and no submitted content is included.

For every `DistillImportError` raised while validating blocks, prefix the message with `source_path` without including event contents. Implement this as one wrapper around block parsing/validation so malformed JSON, schema, role, credential, and limit failures all identify the input; preserve any safe line/event location already in the error. The CLI's source/output collision error must likewise name `arguments.path`.

Update `_split_export_blocks` to accept keyword-only `max_events_per_trajectory: int` and `max_event_bytes: int`, and call it with the `ExternalImporter` constants. In the event-record branch, before appending:

```python
            if len(line.encode("utf-8")) > max_event_bytes:
                raise DistillImportError(
                    f"External event exceeds the size limit at line {line_number}"
                )
            if len(current_events) >= max_events_per_trajectory:
                raise DistillImportError(
                    f"External trajectory exceeds the event count limit at line {line_number}"
                )
            current_events.append(line)
```

Keep whole-file parsing and duplicate-ID validation before the caller writes output. Do not alter the accepted TADP/1 event-type allowlist or credential/private-field rejection.

- [ ] **Step 4: Run importer and schema tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_distill_importers.py tests/test_distill_schema.py -q
```

Expected: all existing compatibility, authorization, redaction-boundary, and newly added limit tests pass.

- [ ] **Step 5: Commit**

```powershell
git add distill/importers.py tests/test_distill_importers.py
git commit -m "feat: enforce agent trajectory import limits"
```

### Task 3: Adding CLI source-model provenance and failure checks

**Files:**
- Modify: `distill/cli.py`
- Test: `tests/test_distill_cli.py`

**Interfaces:**
- `import-external` adds optional `--source-model-id` and `--source-model-revision` string arguments.
- The command passes both values to `ExternalImporter.import_file`; omitting both preserves current behavior.
- CLI output continues to report only the number of imported records and output path; errors continue to return status 2 without echoing trajectory contents. Source/output collisions identify the input path.

- [ ] **Step 1: Add failing CLI metadata, help, and no-partial-output tests**

Add these tests to `tests/test_distill_cli.py`:

```python
def test_import_external_help_lists_optional_model_provenance(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["import-external", "--help"])

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--source-model-id" in help_text
    assert "--source-model-revision" in help_text


def test_import_external_persists_source_model_provenance(capsys, tmp_path):
    source = (
        Path(__file__).parent
        / "fixtures"
        / "distill"
        / "authorized-external-export.jsonl"
    )
    output = tmp_path / "raw" / "claude.jsonl"

    result = cli.main(
        [
            "import-external",
            str(source),
            "--source-name",
            "claude",
            "--license-or-authorization",
            "Synthetic fixture authorization.",
            "--authorized-by",
            "test-operator",
            "--source-model-id",
            "provider/model",
            "--source-model-revision",
            "revision-1",
            "--output",
            str(output),
        ]
    )

    imported = parse_trajectory_jsonl(output.read_text(encoding="utf-8"))
    assert result == 0
    assert imported.provenance.source_model_id == "provider/model"
    assert imported.provenance.source_model_revision == "revision-1"
    assert "Imported 1 authorized trajectory record(s)" in capsys.readouterr().out


def test_import_external_failure_does_not_write_output_or_echo_secret(
    capsys, tmp_path
):
    source = tmp_path / "credential.jsonl"
    lines = (
        Path(__file__).parent
        / "fixtures"
        / "distill"
        / "authorized-external-export.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    secret = "Bearer synthetic-token-value-123456"
    secret = "Bearer " + "synthetic-" + "token-value-123456"
    lines[1] = lines[1].replace("Add a safe check.", secret)
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = tmp_path / "raw" / "rejected.jsonl"

    result = cli.main(
        [
            "import-external",
            str(source),
            "--source-name",
            "claude",
            "--license-or-authorization",
            "Synthetic fixture authorization.",
            "--authorized-by",
            "test-operator",
            "--output",
            str(output),
        ]
    )

    assert result == 2
    assert not output.exists()
    error_text = capsys.readouterr().err
    assert str(source) in error_text
    assert secret not in error_text


def test_import_external_source_output_collision_identifies_input(capsys, tmp_path):
    source = tmp_path / "external.jsonl"
    source.write_bytes(
        (
            Path(__file__).parent
            / "fixtures"
            / "distill"
            / "authorized-external-export.jsonl"
        ).read_bytes()
    )
    original_bytes = source.read_bytes()

    result = cli.main(
        [
            "import-external",
            str(source),
            "--source-name",
            "claude",
            "--license-or-authorization",
            "Synthetic fixture authorization.",
            "--authorized-by",
            "test-operator",
            "--output",
            str(source),
        ]
    )

    assert result == 2
    assert source.read_bytes() == original_bytes
    assert str(source) in capsys.readouterr().err
```

Add `parse_trajectory_jsonl` to the `distill.schema` import in `tests/test_distill_cli.py`.

- [ ] **Step 2: Run the focused CLI tests and confirm they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_distill_cli.py -q
```

Expected: help omits the new flags and argparse rejects the metadata invocation; existing CLI tests remain green.

- [ ] **Step 3: Implement optional CLI arguments**

In `_build_parser`, add:

```python
import_external.add_argument("--source-model-id")
import_external.add_argument("--source-model-revision")
```

In the `import-external` dispatch branch, pass `source_model_id=arguments.source_model_id` and `source_model_revision=arguments.source_model_revision` to `ExternalImporter.import_file`. Change the source/output collision error to include `arguments.path`. Do not make either flag required and do not relax the existing source name, license/authorization, or authorized-by requirements.

- [ ] **Step 4: Run importer and CLI tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_distill_cli.py tests/test_distill_importers.py -q
```

Expected: optional metadata round-trips through normalized JSONL; missing authorization still fails; rejected credentials do not create output or appear in stderr.

- [ ] **Step 5: Commit**

```powershell
git add distill/cli.py tests/test_distill_cli.py
git commit -m "feat: expose agent model provenance in import CLI"
```

### Task 4: Document TADP/1 and verify the integrated distillation path

**Files:**
- Modify: `README.md`
- Test: `tests/test_distill_schema.py`
- Test: `tests/test_distill_importers.py`
- Test: `tests/test_distill_cli.py`
- Test: `tests/test_distill_curator.py`

**Interfaces:**
- README protocol documentation matches `ExternalImporter`, `ImportProvenance`, and the actual `import-external` flags.
- Existing curation and training requirements remain unchanged.

- [ ] **Step 1: Recheck multiple-block compatibility**

The multi-block regression is added in Task 2 before importer changes. Confirm that it passes:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_distill_importers.py::test_external_import_accepts_multiple_unique_trajectories -q
```

Expected: PASS for two distinct version-1 blocks with the same whole-file source hash.

- [ ] **Step 2: Update the README protocol and usage section**

In `README.md` under “Optional local distillation and model evolution,” document:

1. TADP/1 is the current schema-version-1 header followed by ordered, timezone-aware events.
2. External event types are `task`, `verification`, `user_message`, `assistant_message`, and `tool_message`; hidden reasoning, system/developer prompts, credentials, and unknown fields are rejected.
3. `--source-name` identifies the agent/provider; `--source-model-id` and `--source-model-revision` are optional and revision requires model ID.
4. Provider exports must be explicitly supplied or locally converted; no provider sessions are scraped and import does not upload, execute content, curate, or train automatically.
5. The importer's exact limits: 64 MiB per source file, 1,000 events per trajectory, and 64 KiB per encoded event.
6. The existing curate command remains responsible for admitting only success + verification pass + evaluation pass trajectories.

Replace the existing sample command with this accurate optional-provenance example:

```powershell
python -m distill.cli import-external .\authorized-export.jsonl `
  --source-name claude `
  --source-model-id provider/model `
  --source-model-revision pinned-revision `
  --license-or-authorization "Authorized for local fine-tuning under license X" `
  --authorized-by "operator" `
  --output distill/data/raw/claude-export.jsonl
```

- [ ] **Step 3: Run all directly affected tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_distill_schema.py tests/test_distill_importers.py tests/test_distill_cli.py tests/test_distill_curator.py -q
```

Expected: all affected tests pass, including legacy schema-v1 imports, strict external validation, model provenance, limits, CLI safety, and curator eligibility.

- [ ] **Step 4: Check the final diff and whitespace**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only the planned importer/schema/CLI implementation, affected tests, and README changes appear.

- [ ] **Step 5: Commit**

```powershell
git add README.md tests/test_distill_schema.py tests/test_distill_importers.py tests/test_distill_cli.py tests/test_distill_curator.py
git commit -m "feat: finalize TADP agent trajectory import docs and tests"
```

## Self-review

- **Spec coverage:** provenance, legacy compatibility, strict observable-only validation, authorization, hashing, file/event limits, atomic no-partial-write behavior, CLI usage, curation eligibility, and README guidance are assigned to Tasks 1–4.
- **Scope:** this is one local import-protocol subsystem; it does not split provider adapters or training into independent projects.
- **Compatibility:** existing schema-version-1 records default both optional model identity fields to `None`; existing runtime exports and manifests continue to validate.
- **Type consistency:** `source_model_id` and `source_model_revision` use the same names in Pydantic provenance, importer keyword arguments, CLI options, tests, and README.
