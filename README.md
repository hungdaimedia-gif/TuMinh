# TuMinh — autonomous coding agent starter

TuMinh is a bounded Python coding agent starter. It reads one local task by
default (or optionally an open GitHub Issue), plans the work, uses registered
workspace and allowlisted command tools, verifies the result, records a
reflection and local sanitized trajectory, then commits verified changes on a
dedicated `tuminh/task-*` branch. It never merges automatically. Pull requests
are opt-in.

The restrictions below are application-level safeguards, **not an operating
system security sandbox**. Run TuMinh in a disposable clone or a workspace whose
contents you have backed up.

## Requirements and setup

Use Python 3.11 or newer, Git, and an OpenAI-compatible model endpoint. GitHub
CLI (`gh`) is optional unless using GitHub Issues or creating pull requests.

### PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

If PowerShell blocks virtual-environment activation, use
`.venv\Scripts\python.exe` instead of activating it.

### macOS, Linux, or WSL

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` and set `MODEL_API_KEY` for the model endpoint. Keep `.env` local;
it is ignored by Git. `python main.py --help`, unit tests, and deterministic
offline evaluations do not need model or GitHub credentials.

## Configure and run

The default source is the local `tasks/tasks.json` queue:

```powershell
python main.py --once
```

`--once` is the default, so `python main.py` also runs at most one task. To poll
for a bounded number of tasks:

```powershell
python main.py --poll
```

`MAX_POLL_CYCLES` and `POLL_INTERVAL_SECONDS` bound the polling loop. A failed
task returns a nonzero exit code; polling continues to the configured cycle
limit and ultimately returns nonzero if any task failed.

Important `.env` settings:

| Setting | Purpose |
| --- | --- |
| `MODEL_API_KEY` | Required only when constructing the live model client. |
| `MODEL_BASE_URL` | OpenAI-compatible endpoint; defaults to OpenAI. |
| `MODEL_NAME` | Provider model name; `tuminh-active` requires the optional distillation registry phase. |
| `WORKSPACE_ROOT` | Existing local Git repository TuMinh may edit. |
| `TASK_SOURCE` | `local` by default, or `github`. |
| `TASKS_FILE` | Local JSON queue path, default `tasks/tasks.json`. |
| `GITHUB_REPOSITORY` | Optional `owner/repository` for `gh`; without it, `gh` uses the current repository. |
| `CREATE_PULL_REQUEST` | `false` by default. When `true`, push the verified commit and request a PR after task success. |
| `MAX_STEPS`, `MAX_CORRECTIONS`, `MAX_RETRIES` | Positive bounds for model actions, verification repairs, and transient model retries. |
| `COMMAND_TIMEOUT_SECONDS` | Per-subprocess timeout. |
| `ALLOWED_COMMANDS` | JSON array of argv prefixes; never shell command strings. |

The default command allowlist includes focused pytest/eval commands and the Git
and `gh` operations used internally. The model-facing command tool is further
restricted: it cannot invoke Git or GitHub commands directly. Verification
commands cannot stage, commit, push, switch branches, or create PRs. File access
is confined to the resolved workspace, and subprocesses run with `shell=False`
and a timeout.

## Local task queue

Each queue item is validated against this shape:

```json
[
  {
    "id": "starter-document-limit",
    "title": "Document the configured execution-step limit",
    "description": "Inspect agent/config.py and create reports/execution-step-limit.md with one sentence stating the default MAX_STEPS value. Do not change other files.",
    "acceptance_criteria": [
      "The report states the configured maximum execution steps.",
      "No unrelated files are modified.",
      "Run python -m pytest tests/test_config.py -q successfully."
    ],
    "source": "local",
    "state": "pending"
  }
]
```

The tracked queue file is never rewritten. Completed/failed results are stored
in ignored `memory/task_state.json`, so they are not silently retried. To
requeue a task, back up that file and remove only that task's entry. Keep the
queue item in `pending` state.

Before executing file-writing tools, TuMinh creates a task-specific branch from
the current base branch. It verifies the task, runs pytest and deterministic
evals if agent code changed, stages only the task's unambiguous changes, and
commits only after all gates pass. Existing unrelated work is preserved and
never staged. If a task fails after modifying files, inspect those changes
manually before starting another task. After a one-shot run, return to your
base branch before starting another independent task:

```powershell
git status --short
git switch <your-base-branch>
```

TuMinh never merges branches.

## Optional GitHub Issues and pull requests

Install and authenticate GitHub CLI (`gh auth login`), then set:

```dotenv
TASK_SOURCE=github
GITHUB_REPOSITORY=owner/repository
```

TuMinh reads open Issues through `gh issue list`; processed issue IDs are tracked
in the same ignored runtime state file. Local mode does not call `gh` and does
not need GitHub credentials.

Pull requests are disabled by default. To opt in, set
`CREATE_PULL_REQUEST=true`. After task verification, self-evaluation (when
applicable), and a local commit succeed, TuMinh pushes that task branch and
calls `gh pr create`. It never creates a PR for a failed or uncommitted task.
The repository needs an `origin` remote and suitable GitHub permissions.
If pushing or PR creation fails, the verified task remains complete and an
ignored `memory/pull_request_state.json` records the pending publication. The
next `--once` or `--poll` invocation retries that PR before ingesting another
task; TuMinh checks for an existing open PR first to avoid duplicates.

## Reflection and skill proposals

Outcome reflections are stored locally in `memory/reflection.json`; configured
secret values are redacted, and full command output is omitted. Generated rules
remain inert under `skills/proposals/`. A deterministic offline evaluation and
an explicit human confirmation are both required before activation:

```powershell
python main.py approve-skill-proposal skills/proposals/<proposal>.md
```

Active rules belong in `skills/approved/`. TuMinh does not activate proposals
from an agent tool call.

## Trajectories and data boundaries

Sanitized event journals are written locally to
`memory/trajectories/<task-id>.jsonl` and ignored by Git. Events include
observable task, plan, emitted action, tool I/O, diff, verification, and concise
reflection data. The journal rejects hidden reasoning/system/developer prompt
fields, redacts configured secrets and common credential formats, and enforces
event/diff size limits. It does not upload data or train a model automatically.

The later, optional import/curation/training workflow is specified separately
in [`docs/superpowers/plans/2026-10-05-tuminh-distillation.md`](docs/superpowers/plans/2026-10-05-tuminh-distillation.md).
Only explicitly exported, authorized, observable records may enter that
workflow; external provider sessions are never scraped.

## Optional local distillation and model evolution

Distillation is a separate, explicit local workflow. Normal runtime
installation does **not** install training dependencies, train a model,
deploy a candidate, or change the active model. Keep source exports,
curated datasets, adapters, registry state, and benchmark reports local and
access-controlled. External data must be lawfully authorized before import;
the importer requires a source name, authorization/license declaration, and
the person authorizing it. TuMinh does not scrape provider sessions or collect
hidden reasoning.

The base runtime and the export, import, curation, registry, purge, and CLI
help workflows are CPU-only. Benchmarking can also run on CPU when the
explicit local evaluator supports it. Only optional QLoRA training needs the
separate compatible CUDA environment described below.

Set `DISTILL_REGISTRY_PATH` to the local registry file if you do not want its
default, `distill/models/registry.json` under `WORKSPACE_ROOT`. Each command
also accepts `--registry` where appropriate. Start by explicitly recording
the baseline tag and its exact model identity and pinned revision:

```powershell
python -m distill.cli init-active `
  --model-tag baseline-v1 `
  --baseline-model-id authorized/base-model `
  --baseline-model-revision pinned-revision-sha
```

Runtime processes resolve `MODEL_NAME=tuminh-active` through this registry
when the model client is constructed at process startup. Resolution does not
rewrite `.env` and does not fall back if the registry is missing or
uninitialized. For a local Ollama endpoint, configure:

```dotenv
MODEL_BASE_URL=http://localhost:11434/v1
MODEL_NAME=tuminh-active
DISTILL_REGISTRY_PATH=distill/models/registry.json
```

The `baseline-v1` tag must be available to the configured local model endpoint.
Promotions change the registry pointer only; the next runtime process resolves
the new tag during startup.

Export a trajectory explicitly. Runtime records without both passing
verification and eval evidence are excluded from training curation:

```powershell
python -m distill.cli export-runtime `
  --journal-root memory/trajectories `
  --task-id task-123 `
  --output distill/data/raw/task-123.jsonl
```

External import requires explicit authorization metadata; it validates the
observable schema and rejects hidden reasoning fields and credentials:

```powershell
python -m distill.cli import-external .\authorized-export.jsonl `
  --source-name approved-source `
  --license-or-authorization "Authorized for local fine-tuning under license X" `
  --authorized-by "operator@example.test" `
  --output distill/data/raw/authorized-source.jsonl
```

Curate one or more validated trajectory exports. The configured
`MODEL_API_KEY` from the process environment or local `.env`, plus common
credential patterns, are redacted before writing the dataset and its adjacent
manifest:

```powershell
python -m distill.cli curate `
  --input distill/data/raw/task-123.jsonl `
  --input distill/data/raw/authorized-source.jsonl `
  --output distill/data/curated/dataset.jsonl
```

Training is optional and requires an authorized, pinned base model, a curated
dataset with passing verification/eval evidence, a compatible CUDA/Linux or
WSL2 environment, and the separate training stack. Install it only in that
training environment:

```powershell
python -m pip install -r distill/requirements-training.txt
python -m distill.cli train `
  --config distill/config.example.toml `
  --dataset distill/data/curated/dataset.jsonl
```

Copy and edit `distill/config.example.toml` first; use an authorized model
license declaration, exact revision, and ignored candidate output path. The
command prints the candidate manifest path. Its benchmark ID is reserved and
remains pending: training neither benchmarks, deploys, nor promotes.
`rtx_3060_ti_8gb` is an optional conservative profile, not a VRAM guarantee.

Deploy explicitly. Local llama.cpp conversion and Ollama are only invoked by
this command; candidate registration occurs only after the deployment service
successfully converts, installs, and verifies the candidate:

```powershell
python -m distill.cli deploy `
  --manifest distill/models/candidates/candidate-example/candidate-manifest.json `
  --adapter distill/models/candidates/candidate-example `
  --converter-script C:\src\llama.cpp\convert_lora_to_gguf.py `
  --base-model-path C:\models\authorized-base
```

Benchmark an installed candidate with the bundled versioned eval case set, or
provide a local `--eval-set`. A local evaluator plugin is required because the
bundled cases contain descriptions and safety labels, not objective
model-input/expected-output pairs. Supply `--evaluator module:factory`; the
factory returns an object implementing:

```python
def evaluate(
    model_name,
    cases,
    *,
    tool_limits,
    decoding_config,
) -> ModelScore:
    ...
```

`model_name` is the registered local model tag (the current active tag for the
baseline and the candidate ID for the candidate). The evaluator owns the
actual objective task/evidence mapping, maps those tags to its local runtime,
and returns `ModelScore` values with the exact model ID and pinned revision
from the registry/candidate manifest. Benchmarking runs it on the baseline and
candidate with shared keyword-only settings and writes a typed report. It does not
invent self-scoring evidence or infer benchmark success from training. The
candidate's reserved benchmark ID is used as-is:

```powershell
python -m distill.cli benchmark `
  --candidate-manifest distill/models/candidates/candidate-example/candidate-manifest.json `
  --evaluator my_local_evaluator:build `
  --output distill/reports/candidate-example-benchmark.json
```

Promotion displays baseline/candidate scores and safety outcomes, then
requires an explicit affirmative interactive confirmation and approver name.
The registry independently verifies report completeness, exact baseline and
candidate identity, pass-rate non-regression, and all safety-critical results.
An ineligible report cannot be promoted:

```powershell
python -m distill.cli promote `
  --candidate-id candidate-example `
  --report distill/reports/candidate-example-benchmark.json
```

Rollback is a separate explicit registry operation and preserves installed
versions:

```powershell
python -m distill.cli rollback
```

Purge removes only the raw file whose filename and content match the exact
source SHA-256 and removes rows linked to that hash from curated datasets:

```powershell
python -m distill.cli purge `
  --source-sha256 <64-character-lowercase-sha256> `
  --raw-root distill/data/raw `
  --curated-root distill/data/curated
```

All commands operate on explicit local files and paths. CLI help, unit tests,
and deterministic evals use fakes and do not contact a model provider,
download models, invoke Ollama, or require training dependencies.

## Offline checks

```powershell
python -m pytest -q
python -m evals.run
python main.py --help
```

These checks use fakes and temporary workspaces; they do not call a model
provider, GitHub, CUDA, or Ollama.
