# Tự Minh Multi-Agent Distillation: Agent Trajectory Protocol

**Date:** 2026-10-05
**Status:** Draft for user review
**Scope:** Blueprint for importing authorized, observable agent trajectories into the existing local distillation workflow.

## Context

Tự Minh already has an explicit local distillation pipeline:

1. `ExternalImporter` validates newline-delimited JSON exports, requires operator-supplied source and authorization metadata, rejects hidden/private fields and credential-like values, and computes a SHA-256 source hash.
2. `Curator` extracts observable messages, redacts configured secrets, and admits only successful trajectories with passing verification and evaluation.
3. `Trainer` consumes the curated dataset only when explicitly invoked.

The existing external format has a version-1 header and observable events. This design formalizes that format as the **Tự Minh Agent Trajectory Protocol (TADP/1)** and makes it practical for Claude, Antigravity, DeepSeek, and other agents to submit compatible local exports. Provider sessions must not be scraped, uploaded, or queried automatically.

## Goals

- Define one provider-neutral, versioned JSONL interchange contract.
- Preserve explicit operator authorization and auditable source provenance.
- Admit only observable user, assistant, and tool messages; never hidden reasoning, system/developer prompts, or credentials.
- Reuse the existing importer, curator, and training gates wherever possible.
- Keep import local, explicit, atomic, and testable without provider credentials.

## Non-goals

- Live provider API connections, session scraping, browser automation, or remote uploads.
- Automatic collection, curation, training, deployment, promotion, or active-model changes.
- Authenticating the identity of an upstream agent cryptographically.
- Provider-specific parsers in TADP itself. Any conversion from a provider's own export format belongs in a separate local adapter.

## Protocol and data model

TADP/1 formalizes the existing `schema_version: 1` JSONL representation rather than introducing a new incompatible header. A file contains one or more trajectory blocks. Each block starts with a header record and is followed by its event records:

- The header declares `record_type: "header"`, `schema_version: 1`, `trajectory_id`, `task_id`, `task_outcome`, `verification_passed`, and nullable `eval_passed`.
- Events declare `schema_version: 1`, the matching `task_id`, contiguous one-based `sequence_number`, timezone-aware `timestamp`, an allowed `event_type`, and its data.
- TADP/1 external events are limited to `task`, `verification`, `user_message`, `assistant_message`, and `tool_message`. Message events carry exactly one `message` object with the matching `user`, `assistant`, or `tool` role and non-empty content.
- The source agent/provider is identified by the required operator-supplied `--source-name` value (for example `claude`, `antigravity`, or `deepseek`). Optional model ID and revision metadata may be supplied at import time. Authorization/license declaration and authorizing operator remain explicit import-time metadata; they are not trusted from the submitted trajectory.
- The importer computes SHA-256 over the exact source bytes and attaches it to every imported trajectory's provenance. The hash supports integrity and traceability, not source authentication.

Existing version-1 exports remain valid. New optional provenance fields, if added for model identity, must not make existing exports or stored manifests invalid. Unknown fields remain rejected. Future incompatible protocol changes require a new schema version and an explicit compatibility decision.

## Data flow

```text
Agent output or local adapter
        |
        v
TADP/1 JSONL file
        |
        v
Explicit `import-external` + operator authorization
        |
        v
Whole-file schema, safety, size, sequence, and provenance validation
        |
        v
Atomically written local raw trajectory export
        |
        v
Existing curator: redact secrets and require success + verification + eval
        |
        v
Existing explicit training command
```

Importing does not execute tool payloads or message content. Imported raw data remains local and is not sent to an agent, provider, or remote service. A trajectory with missing or failing evaluation evidence may be retained as an authorized raw import, but the curator must exclude it from training data.

## Validation and failure behavior

Validation is fail-closed for the complete input file: malformed JSONL, empty lines, invalid or unknown fields, prohibited private fields, credential-like values, incorrect role/event combinations, mismatched task IDs, non-contiguous sequence numbers, duplicate trajectory IDs within a file, missing authorization, or size-limit violations reject the file before any output is written.

Limits for the first implementation:

- Maximum source file size: 64 MiB.
- Maximum events per trajectory: 1,000.
- Maximum encoded event size: 64 KiB, aligned with the runtime journal's existing event limit.

Errors identify the input and, where available, the line or event index and failure category; they must not echo rejected message contents or secrets. Valid imports are written atomically using the existing safe-write pattern. The source file cannot be overwritten by its own import output.

## CLI and local adapters

Keep the existing command and its explicit authorization parameters:

```powershell
python -m distill.cli import-external .\authorized-export.jsonl `
  --source-name claude `
  --license-or-authorization "Authorized for local fine-tuning under license X" `
  --authorized-by "operator" `
  --output distill/data/raw/claude-export.jsonl
```

Use the same command with the appropriate source label for Antigravity, DeepSeek, and other compatible producers. Optional model ID/revision flags may capture model provenance when it is known. A local adapter may convert a provider's explicitly exported data to TADP/1, but must not fetch or scrape sessions. Adapter output is untrusted and must pass the same importer validation.

The supported training invocation remains `python -m distill.cli train --config ... --dataset ...`; training additionally requires a valid authorized model configuration and a curated dataset. Importing an agent trajectory does not satisfy those prerequisites by itself.

## Implementation boundaries

The first implementation should be limited to:

- `distill/schema.py`: protocol-compatible optional provenance fields and strict validation as needed.
- `distill/importers.py`: TADP limits and full-file validation without partial writes.
- `distill/cli.py`: optional model provenance arguments and safe import reporting, retaining `import-external`.
- Importer/schema tests and synthetic fixtures; no real provider transcripts or secrets.
- `README.md`: TADP/1 shape, local import flow, authorization, and adapter boundary.

No model-training, deployment, registry, runtime collection, or provider API modules should change as part of this blueprint's first implementation.

## Verification plan

Tests must demonstrate:

- Existing authorized schema-v1 fixtures continue to import.
- Valid TADP/1 records from distinct provider labels retain operator authorization and source hashes.
- Unknown/private fields, credentials, invalid roles, mismatched IDs, bad timestamps, malformed framing, duplicate IDs, and size-limit violations reject without partial output.
- Imported data is not executed and its content is not included in error output.
- Curation retains its existing rules: only successful, verified, evaluation-passing trajectories become training rows; missing evaluation evidence is excluded.
- CLI help and README examples match the supported `import-external` invocation.

## Open implementation detail

The current repository does not include provider-specific exports or adapters. Their field mapping and any provider-specific lawful-use requirements should be documented and reviewed independently when an actual, authorized export format is available; TADP/1 remains the stable boundary.
