# TuMinh Distillation and Model Evolution Engine — Architecture Design

**Status:** Updated design; review pending
**Date:** 2026-10-05

This is the authoritative design for the distillation phase. The earlier
unreviewed `2026-10-05-tuminh-distillation-and-evolution-spec.md` is preserved
as source material; this design supersedes it where they differ, particularly
on data collection, hardware guarantees, promotion, and runtime configuration.

## Goal

Define a separate, optional `distill/` phase that converts authorized, observable
coding-agent trajectories into a curated dataset, optionally fine-tunes a local
candidate model, evaluates it against the currently promoted model, and
promotes or rolls back versioned Ollama models only after explicit human
approval.

## Design decisions

- Build this as a separate phase after the autonomous runtime and its event
  journal/evals are working; keep CUDA and training dependencies out of the
  runtime installation.
- Accept TuMinh runtime exports and manually supplied external exports only
  when the user declares that they have the right to use them.
- Store task, plan, emitted responses/actions, tool I/O, diffs, verification
  results, and concise reflections. Never ingest hidden chain-of-thought,
  provider system prompts, credentials, or environment values.
- Only successful trajectories with passing verification/evals can enter the
  curated training set.
- Make local QLoRA training optional and configurable. RTX 3060 Ti 8GB is an
  optional profile, not a promise of fixed VRAM usage or throughput.
- Benchmark a candidate against the current model on the same pinned eval
  suite. Require candidate pass rate at least equal to baseline and 100% pass
  on safety-critical evals before it is eligible for promotion.
- Require an explicit human confirmation to promote. Keep versioned model
  tags and a tested rollback operation.
- Keep raw exports, curated datasets, checkpoints, adapters, and model weights
  local and out of Git.

## Repository layout

```text
distill/
  __init__.py
  cli.py                 explicit import/export/curate/train/evaluate/promote/rollback commands
  schema.py              trajectory, provenance, dataset, and candidate-manifest schemas
  importers.py           strict runtime JSONL and user-provided export importers
  curator.py             validation, redaction, eligibility checks, and dataset creation
  trainer.py             optional QLoRA adapter training
  benchmark.py           baseline/candidate comparison using pinned eval cases
  registry.py            candidate/promoted versions and rollback metadata
  deploy.py              optional GGUF conversion and Ollama model-tag operations
  config.example.toml    model, training, hardware-profile, and eval configuration
  data/
    raw/                 ignored local exports
    curated/             ignored local datasets and manifests
  models/
    registry.json        local-only active/candidate/previous model pointers
    candidates/          ignored adapters, GGUFs, and checkpoints
```

The runtime phase owns `memory/trajectories/` and emits versioned, redacted
JSONL events. The distillation CLI reads those events only when an explicit
export command is run. The existing runtime remains usable with its configured
OpenAI-compatible endpoint when training or Ollama is unavailable.

## Data sources and trajectory contract

### TuMinh runtime exports

Each exported trajectory identifies its schema version, task ID, outcome,
verification result, and ordered events. Allowed event types are task,
generated-plan, emitted-decision, tool-call, redacted-tool-result, patch,
verification, and concise-reflection. Events must not contain hidden/private
reasoning, system/developer prompts, API credentials, environment contents, or
unredacted secret-bearing output. The importer rejects unknown fields and
prohibited reasoning fields instead of silently copying them.

The export is a local file and requires an explicit CLI operation, for example:

```text
python -m distill.cli export-runtime --task-id TASK_ID --output PATH
```

The runtime never uploads trajectories or turns them into training data by
itself.

### User-provided external exports

External exports are manually supplied files, not scraped or harvested from
provider sessions. Import requires source and license/authorization metadata.
The importer maps only observable user/assistant/tool messages and relevant
verification evidence into the trajectory schema; it rejects fields such as
`thought`, `thinking`, `analysis`, or equivalent private-reasoning fields.
Imports with missing provenance, invalid schema, or unclear authorization are
rejected with an actionable error.

## Curation and dataset generation

`curator.py` validates records, redacts configured secrets, rejects forbidden
fields, normalizes event ordering, and checks task outcome. Only successful
tasks whose verification and required eval results passed are eligible for
training. Failed task reflections remain in runtime memory for operational
learning but are not automatically mixed into supervised training data.

An explicit curation command writes versioned JSONL records plus a manifest
containing source IDs, authorization metadata, curation version, redaction
counts, and source hashes. It does not copy raw logs into Git. Dataset rows use
only standard `user`, `assistant`, and `tool` roles; no `thought` role or
private-reasoning field is allowed. Before writing an output file, validate the
whole batch so malformed input does not produce a success-shaped partial
dataset.

## Optional local training

The base project install includes no CUDA/Unsloth packages. A separately
documented training extra/environment supplies a supported Unsloth, PyTorch,
CUDA, and model-conversion stack on compatible Linux or WSL2 systems. CPU-only
users can still import, curate, and benchmark fixtures without installing it.

The trainer uses configurable base-model ID/revision, model license metadata,
quantization, sequence length, batch size, gradient accumulation, LoRA rank,
and output directory. QLoRA produces a candidate adapter; optional conversion
creates a GGUF and a versioned Ollama candidate tag. The RTX 3060 Ti 8GB profile
starts with conservative settings and measures actual peak VRAM. It may lower
sequence length or batch settings, or stop with an explicit out-of-memory
failure. No fixed VRAM or tokens-per-second guarantee is made.

Checkpoints and weights stay in ignored local paths. Training failure leaves
the promoted model and previous registry entry unchanged.

## Benchmark, promotion, and rollback

The benchmark runs the same pinned eval cases against the active baseline and
candidate with the same tools, limits, and decoding settings. It records model
IDs/revisions, dataset hash, eval-set version, pass rate, safety-critical
results, software/hardware versions, and resource measurements.

A candidate is eligible only when:

1. Its aggregate pass rate is at least the active model's pass rate on the same
   eval set.
2. Every safety-critical eval passes.
3. Required model, dataset, provenance, and benchmark metadata are present.

Eligibility does not activate a candidate. The explicit `promote` command
displays the comparison and requires an affirmative human response. It copies
the candidate to a versioned Ollama tag, then atomically updates the active
version pointer in `distill/models/registry.json`, keeping the prior active
version for rollback. For local Ollama use, the user configures the runtime
once with `MODEL_BASE_URL=http://localhost:11434/v1` and
`MODEL_NAME=tuminh-active`; at process startup the runtime resolves that
symbolic name to the versioned tag in the registry. Promotion does not rewrite
`.env` or switch a running process. A `rollback` command atomically restores
the previous registry pointer; all versioned model tags remain available.
Failed promotion or registry writes leave the prior pointer and running model
usable.

## Error handling and data protection

Use typed errors for unsupported exports, missing provenance, schema
violations, redaction failures, training/deployment failures, and failed evals.
Report rejected record IDs and reasons without logging their unredacted
contents. Do not swallow incomplete writes, CUDA failures, Ollama errors, or
registry-update failures. Write datasets/manifests and registry updates through
temporary files and atomic replacement where applicable.

Ignore `distill/data/raw/`, `distill/data/curated/`, checkpoints, adapters,
GGUFs, and runtime registry state in Git. Provide an explicit purge command for
raw and curated records by source ID/hash; do not silently delete data during
curation. Commit only source code, synthetic
fixtures, and non-sensitive example configuration. The importer records
authorization metadata; it cannot independently establish copyright ownership,
so the operator remains responsible for the declared rights.

## Testing and evaluation

- Unit-test event schemas, unknown/prohibited fields, provenance requirements,
  redaction, successful-task eligibility, and atomic dataset output.
- Test runtime export and external import with synthetic fixtures only; do not
  call provider APIs or collect external sessions in unit tests.
- Test benchmark threshold logic and ensure any failed safety-critical eval
  blocks promotion.
- Test human rejection, missing approval, failed deployment, registry-write
  failure, and rollback using a fake Ollama adapter.
- Run CPU-only checks in normal CI. Run CUDA training and real Ollama integration
  only through explicit local opt-in tests.

## Out of scope

- Harvesting provider sessions, hidden chain-of-thought, system prompts, or
  unprovided third-party data.
- Automatic uploads, training runs, or model promotion.
- Guaranteeing that a specific 7B model fits in 8GB VRAM or reaches a fixed
  inference speed.
- Committing datasets, adapters, checkpoints, or weights to the repository.
- Replacing the runtime model used by a running process without an explicit
  user-controlled restart/configuration.

## Acceptance criteria

1. Runtime trajectories can be explicitly exported as validated JSONL without
   secrets, private reasoning, or system prompts.
2. External imports require declared authorization/provenance and reject
   prohibited reasoning fields and malformed schemas.
3. Curation excludes failed/unverified tasks and produces a dataset manifest
   with source hashes and authorization metadata.
4. Dataset import, curation, benchmark-threshold, promotion-gate, and rollback
   tests pass on CPU without provider/GitHub credentials or network access.
5. Candidate evaluation uses the same pinned suite and records enough metadata
   to reproduce the comparison.
6. Promotion is blocked unless aggregate pass rate is non-regressing, all
   safety-critical evals pass, and a human confirms.
7. A failed training, deployment, promotion, or registry write leaves the
   currently promoted model available.
8. RTX 3060 Ti 8GB remains an optional measured profile; no fixed memory or
   throughput claim is an acceptance requirement.
