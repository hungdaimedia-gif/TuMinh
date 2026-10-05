Task 2 report — harden TADP/1 import

Scope
- Task: enforce import limits and provenance safety for external trajectory imports.
- Worktree: C:\Users\Admim\TuMinh\.worktrees\tadp-agent-trajectory
- Brief: .superpowers/sdd/2026-10-05-multi-agent-distillation/task-2-brief.md
- Spec: docs/superpowers/specs/2026-10-05-multi-agent-distillation-design.md

Files changed
- distill/importers.py
- tests/test_distill_importers.py

Commit
- c52e6fd
- Message: feat: enforce agent trajectory import limits

RED: failing TDD command and output
Command:
.
.\.venv\Scripts\python.exe -m pytest tests/test_distill_importers.py -q
.
Output summary:
- 5 failed, 13 passed in 0.43s
- Failures were the new Task 2 behaviors: source_model_id/source_model_revision API, _MAX_SOURCE_FILE_BYTES, _MAX_EVENT_BYTES, _MAX_EVENTS_PER_TRAJECTORY, and provenance validation.

GREEN: passing validation command and output
Command:
.
.\.venv\Scripts\python.exe -m pytest tests/test_distill_importers.py tests/test_distill_schema.py -q
.
Output summary:
- 47 passed in 0.28s

Implementation notes
- `ExternalImporter` now defines the required limits:
  - `_MAX_SOURCE_FILE_BYTES = 64 * 1024 * 1024`
  - `_MAX_EVENTS_PER_TRAJECTORY = 1_000`
  - `_MAX_EVENT_BYTES = 64 * 1024`
- `import_file` accepts the Task 1 API: `source_model_id` and `source_model_revision` with the exact names from the provenance model.
- Source validation checks `stat().st_size` before reading and `len(source_bytes)` after reading, so oversize files fail before content processing or partial writes.
- `_split_export_blocks` accepts keyword-only `max_events_per_trajectory` and `max_event_bytes` and rejects:
  - event lines whose UTF-8 encoding exceeds `_MAX_EVENT_BYTES`
  - trajectories whose current event count is already at the limit before appending the next event
  - duplicate trajectory IDs across the input file
  - malformed headers or malformed JSONL before any result is returned
- Every block is validated before a result is returned, preventing partial success from a mixed file.

Path/error handling and no-content guarantees
- Import errors are prefixed with the source path to identify the failing file while keeping payload content out of the error string.
- Invalid schema, unsafe role/content, credential-like values, malformed JSON, and limit violations are wrapped as `DistillImportError` with file-path prefixes and safe category details such as line numbers or event count; they do not echo submitted message bodies or payload values.
- Provenance validation failures are wrapped as the safe public message: `External import provenance is invalid`.
- The importer preserves the existing safe behavior: no partial writes, no secret echoing, and no content returned in error output.

Deviations
- None. Scope stayed within `distill/importers.py` and `tests/test_distill_importers.py`, as required by the brief.
- `distill/schema.py` was intentionally not edited because Task 1 already supplied the required `source_model_id`/`source_model_revision` provenance fields and validation.

Concerns
- None in the current scoped change. The remaining scope boundary is intentionally narrow to the TADP/1 import hardening and does not alter CLI, provider adapters, schema, or training/deployment code.


## Fix round — provenance error source path

Finding addressed
- Provenance `ValidationError` was converted to the safe generic `DistillImportError` message without identifying the external input path.
- The public text now prefixes that unchanged generic message with `source_path`. Pydantic validation details and supplied values remain hidden.

Files changed
- `distill/importers.py` — prefix the provenance error with the source path.
- `tests/test_distill_importers.py` — assert the path and generic error are present and the supplied revision value is not exposed.
- `.superpowers/sdd/2026-10-05-multi-agent-distillation/task-2-report.md` — appended this fix-round report.

RED
Command (run from the worktree):
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_distill_importers.py::test_external_import_rejects_revision_without_model_id -q
```
Result: 1 failed as expected. The existing message contained `External import provenance is invalid` but did not contain the source path; the new assertion failed on the missing path.

GREEN
Command (run from the worktree):
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_distill_importers.py tests/test_distill_schema.py -q
```
Result: 47 passed in 0.31s.

Commit
- `bd6622f03bb40983bebd218c667e71f283437941` — `fix: include source path in provenance errors`

Concerns
- None.
