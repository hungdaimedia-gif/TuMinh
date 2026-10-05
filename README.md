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

## Offline checks

```powershell
python -m pytest -q
python -m evals.run
python main.py --help
```

These checks use fakes and temporary workspaces; they do not call a model
provider, GitHub, CUDA, or Ollama.
