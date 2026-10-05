# TuMinh Autonomous Coding Agent — Architecture Design

**Status:** Approved design; awaiting review of this written specification
**Date:** 2026-10-05

## Goal

Create a runnable Python starter for TuMinh, a single-agent autonomous coding
assistant that ingests queued work, plans and executes bounded code changes,
runs verification, records outcomes, and can prepare a branch and pull request.
The starter must run against a local JSON queue without GitHub credentials and
must support an OpenAI-compatible model endpoint through configuration.

## Design decisions

- Use a modular, single-process Python application. Do not introduce a
  multi-agent framework or provider-specific orchestration framework.
- Keep model access behind an interface and use an OpenAI-compatible API client
  configured by endpoint, model, and API key.
- Make `tasks/tasks.json` the default task source. Add GitHub Issues as an
  optional source; use the GitHub CLI (`gh`) for optional issue/PR operations.
- Run changes only on a dedicated working branch. Never merge automatically.
- Bound execution by configurable step, retry, and subprocess time limits.
- Treat learned skills/rules as proposals. Require an evaluation and human
  approval before loading a proposal into active behavior.

## Repository layout

```text
agent/       CLI entry point, orchestration loop, model interface, task schema
tools/       workspace file operations, command runner, Git and GitHub adapters
memory/      reflection store (JSON)
tasks/       local JSON task queue and task examples
skills/      approved prompt rules and pending skill/rule proposals
tests/       unit and integration tests with fake model/tool implementations
evals/       deterministic safety and behavior evaluation cases
docs/        architecture and usage documentation
```

The starter targets Python 3.11 or newer and includes a root-level `main.py`,
`.env.example`, dependency metadata, and a README with setup and operating
instructions. Secrets are supplied through the environment and excluded from
version control.

## Components and responsibilities

### Task ingestion

The queue reader loads and validates local JSON tasks. A GitHub Issues adapter
is optional and disabled unless configured. Both sources produce the same
normalized task type, including an identifier, title, description, acceptance
criteria, source, and state.

The planner converts a task into a structured plan containing ordered
subtasks, intended files, and verification commands. Validate the model output
against a schema before execution. Source text is untrusted input and must
never override system policy or tool restrictions.

### Model and execution loop

The model interface accepts the current task, plan, bounded context, and
available tool descriptions, and returns either a structured tool request or a
final response. The orchestrator enforces the ReAct cycle:

1. Parse the task and produce a plan.
2. Request and validate one tool operation.
3. Execute it through the tool registry and return the result to the model.
4. Run the task's configured verification; on failure, permit bounded
   self-correction attempts.
5. Record the outcome and, if successful, prepare a commit and optionally a
   pull request.

Exceeding a configured limit or encountering an unrecoverable error ends the
task as failed, records the reason, and does not create a success-shaped
result. The default CLI mode processes one task; an optional polling mode must
also have a configured interval and resource limits.

### Tools and repository operations

Expose narrow, typed operations for reading files, writing files, listing
workspace files, running approved verification commands, and requesting Git
operations. Resolve and validate every file path against the configured
workspace root; reject path traversal and writes outside that root.

Do not expose an unrestricted shell tool. The command runner accepts only
configured command patterns, captures output, applies timeouts, and reports
non-zero exit status as an error. Git operations must verify the working
branch and a clean/understood change set before committing. PR creation is
optional and uses `gh`; merging is out of scope.

These application-level restrictions reduce accidental damage but are not an
OS security sandbox. Run the agent with least-privilege credentials and in a
disposable or otherwise controlled workspace.

### Memory and self-improvement

Append task outcome reflections to `memory/reflection.json` using a validated
schema. Store task identifiers, outcome, concise lessons, and failure
categories; do not store API keys, environment values, or full secret-bearing
command output.

When the agent identifies a reusable lesson, it may create or update a
proposal under `skills/proposals/`. Proposals are inert and are not added to
the active prompt automatically. Activation requires deterministic evals to
pass and explicit human approval. The agent may not modify its own runtime
policy or silently activate a self-authored rule.

When a task changes the agent itself, run the agent's unit tests and
deterministic eval suite before allowing a commit. A failed test/eval prevents
the commit and is recorded as a failed task.

## Configuration and operation

Configuration comes from environment variables documented in `.env.example`,
including the model API key, base URL, model name, workspace root, execution
limits, allowed commands, and optional GitHub settings. Local queue operation
must not require an API token for GitHub. The README will document installation,
queue format, configuration, one-shot execution, optional polling, and GitHub
setup.

## Error handling and observability

Use explicit validation and typed errors at model, task, tool, subprocess, and
GitHub boundaries. Surface actionable failures in CLI output and structured
logs. Retry only transient model/API failures within a configured bound.
Redact configured secrets from logs; do not silently skip malformed tasks,
failed commands, or reflection write errors.

## Testing and evaluation

- Unit-test task parsing, model response validation, path containment, command
  allowlisting/timeouts, reflection persistence, and Git branch/commit guards.
- Use fakes for the model and external commands so the standard test suite
  needs no API key, GitHub credentials, or network access.
- Add deterministic eval cases for planning shape, tool argument validation,
  failure reporting, untrusted task text, and self-improvement proposal
  approval gates.
- Exercise optional live model and GitHub integrations only through explicit
  opt-in configuration.

## Out of scope

- Automatic merging or unrestricted execution on a user's primary branch.
- Multi-agent coordination, hosted workers, web UI, and production deployment.
- Autonomous activation of skills or edits to runtime policies.
- Guarantees that prompt instructions alone can neutralize malicious issue
  content; protection relies on tool boundaries, validation, least privilege,
  and human review for learned behavior.

## Acceptance criteria

1. A fresh checkout can install dependencies and run the test suite without
   credentials or network access.
2. A local task can be loaded, planned, executed using fake tools, verified,
   and recorded in reflection storage.
3. The configured model endpoint can be changed without changing orchestration
   code.
4. File operations reject paths outside the workspace, and command execution
   rejects commands outside the allowlist or beyond timeout.
5. Git changes remain on a dedicated branch; failed verification blocks a
   commit, and no workflow merges automatically.
6. Self-improvement produces an inert proposal and cannot activate it without
   eval success and human approval.
7. The README and `.env.example` explain local queue setup and optional GitHub
   operation.
