# Copilot Instructions for TuMinh

TuMinh is a bounded, Python 3.11+ autonomous coding agent: it reads one task (local JSON queue
or a GitHub Issue), plans with an OpenAI-compatible model, executes via a small allowlisted tool
registry, verifies the result with pytest/evals, records a reflection + sanitized trajectory, and
commits to a dedicated `tuminh/task-*` branch. It never merges and never opens a PR unless
`CREATE_PULL_REQUEST=true`. A separate, optional `distill/` workflow curates trajectories and
trains/benchmarks/promotes local candidate models. **Read `README.md` first** — it is the
authoritative spec for behavior, env vars, and every CLI workflow; this file only adds what a
Copilot session needs to navigate and extend the code safely.

## Setup, build, test

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

- Full test suite: `python -m pytest -q` (pytest config lives in `pyproject.toml`,
  `testpaths = ["tests"]`).
- Single test file: `python -m pytest tests/test_orchestrator.py -q`
- Single test case: `python -m pytest tests/test_orchestrator.py::test_name -q`
- Deterministic offline evals (no model/network calls): `python -m evals.run`
- CLI smoke check: `python main.py --help`
- All of the above use fakes/temp workspaces — they never call a live model, GitHub, CUDA, or
  Ollama. Only `distill/cli.py train` needs the separate CUDA stack
  (`pip install -r distill/requirements-training.txt`), and `deploy`/`benchmark` need
  llama.cpp/Ollama — install those only when actually exercising that workflow.
- There is no configured lint/format tool; match existing style (type hints everywhere,
  `from __future__ import annotations`, dataclasses/pydantic models for data, narrow custom
  exceptions per module).

## Architecture

- `main.py` — composition root. `create_application()` wires `Settings` → `CommandRunner`
  (allowlisted argv prefixes) → `Workspace` → `ToolRegistry` → `Orchestrator`, and owns the
  pending-pull-request state machine (`memory/pull_request_state.json`) so a crash between
  "commit succeeded" and "PR created" is retried idempotently on the next run rather than
  duplicated.
- `agent/` — the core loop: `config.py` (env-sourced `Settings`, including the command
  allowlist), `task_queue.py`/`github_source.py` (the two `TaskSource` implementations),
  `planner.py` + `prompts.py` (model-facing planning), `orchestrator.py` (drives
  plan → tool-call loop → `verifier.py` → `self_eval.py` → `reflection.py` → `trajectory.py`),
  `models.py` (the pydantic `Task`/`TaskResult`/`ToolCall`/etc. contracts shared across layers).
- `tools/` — the actual side-effecting operations: `command_runner.py` (the only way any code
  runs a subprocess — always `argv` lists, never shell strings, always `shell=False` with a
  timeout), `git_ops.py`/`github_ops.py` (branch/commit/PR operations, called by `main.py`/
  orchestrator, never exposed to the model directly), `workspace.py` (path-confined file I/O),
  `registry.py` (generic pydantic-schema-validated tool dispatch used by `main.py` to expose
  `read_file`/`write_file`/`list_files`/`run_command` to the model).
- **Three separate command-allowlist tiers**, all derived from one `ALLOWED_COMMANDS` setting —
  don't blur them: the model's `run_command` tool (`_agent_command_prefixes`, strips `git`/`gh`
  entirely), the verifier's command runner (`_verification_command_prefixes`, strips
  `gh` and mutating `git` subcommands like `add`/`commit`/`push`/`switch`/`reset`), and the
  internal `git_ops`/`github_ops` runner (full allowlist, invoked by orchestrated code only).
- `distill/` — independent pipeline: `importers.py` (TADP/1 protocol — strict schema, size caps,
  rejects hidden reasoning/system prompts/credentials) → `curator.py` (admits only trajectories
  with passing verification + eval evidence, redacts secrets) → `trainer.py` (QLoRA, needs CUDA)
  → `deploy.py` (llama.cpp convert + Ollama install) → `benchmark.py` (requires a user-supplied
  `--evaluator module:factory` plugin; the bundled `evals` cases are safety-labeled descriptions,
  not objective pairs) → `registry.py` (tracks baseline/candidate tags; `promote` independently
  re-verifies report completeness/non-regression/safety before flipping the active pointer).
  `MODEL_NAME=tuminh-active` resolves through this registry at process startup only.
- `memory/` — all runtime state (`task_state.json`, `reflection.json`,
  `pull_request_state.json`, `trajectories/*.jsonl`) is git-ignored and treated as the source of
  truth for "was this already done" — the tracked `tasks/tasks.json` queue file itself is never
  rewritten by the agent.
- `skills/proposals/` (generated, inert) vs `skills/approved/` (human-reviewed, active) — nothing
  in `proposals/` takes effect until `python main.py approve-skill-proposal <path>` runs
  deterministic eval **and** gets explicit interactive human confirmation.

## Conventions

- Every cross-boundary payload (tasks, tool calls, results, trajectory events, distill
  manifests/reports) is a pydantic `BaseModel` with `model_config = ConfigDict(extra="forbid")` —
  add fields to the model, don't pass loose dicts.
- Config is only ever read through `Settings.from_env(...)`; don't read `os.environ` elsewhere.
  New settings need validation in `config.py` (the `positive_int` pattern) plus a default and a
  row in the README's settings table.
- Secrets (`MODEL_API_KEY` + common credential patterns) are redacted before anything is written
  to `reflection.json` or a trajectory journal — thread `secret_values` through if you add a new
  writer, don't bypass it.
- File writes that must survive a crash (pending PR state, distill registry) use the
  write-to-temp-file-in-same-dir + `fsync` + `os.replace` pattern in `main.py`/`distill/registry.py`
  — follow it for any new durable state file, and keep such paths asserted inside
  `WORKSPACE_ROOT` (see `_assert_pr_state_path_contained`), as a defense against the model's
  `write_file`/`read_file` tool escaping the workspace.
- Tests are grouped by module under `tests/` (`test_<module>.py`), use the fakes in
  `tests/fakes.py`/`tests/distill_fakes.py` instead of real subprocess/model/network calls, and
  generally construct the unit under test directly rather than going through `main.py` unless the
  test is specifically about CLI wiring (see `test_cli.py`, `test_starter_flow.py`).
