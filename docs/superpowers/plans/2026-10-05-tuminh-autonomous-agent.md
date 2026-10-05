# TuMinh Autonomous Coding Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Python starter that accepts local or GitHub tasks, plans and performs bounded code changes, verifies them, records reflections, and prepares a branch/optional pull request.

**Architecture:** Use a single-process Python 3.11+ application with typed Pydantic domain models, a provider-neutral model protocol backed by an OpenAI-compatible client, and a bounded orchestrator that can invoke only registered tools. Keep queue, filesystem, command execution, Git/GitHub, reflection, and skill proposals behind separate modules; require verification and a dedicated branch before any commit.

**Tech Stack:** Python 3.11+, OpenAI Python SDK (configurable `base_url`), Pydantic 2, python-dotenv, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-tuminh-autonomous-agent-design.md`

## Global Constraints

- The starter targets Python 3.11 or newer.
- Make `tasks/tasks.json` the default task source.
- Run changes only on a dedicated working branch. Never merge automatically.
- Bound execution by configurable step, retry, and subprocess time limits.
- Treat learned skills/rules as proposals. Require an evaluation and human approval before loading a proposal into active behavior.
- Do not expose an unrestricted shell tool.
- These application-level restrictions reduce accidental damage but are not an OS security sandbox.
- Local queue operation must not require an API token for GitHub.
- Use fakes for the model and external commands so the standard test suite needs no API key, GitHub credentials, or network access.
- A failed test/eval prevents the commit and is recorded as a failed task.

---

## File map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Python package metadata, runtime dependencies, pytest configuration |
| `.gitignore` | Exclude `.env`, bytecode, virtualenvs, and generated reflection/runtime state |
| `.env.example` | Document model, workspace, execution-limit, command-allowlist, and optional GitHub settings |
| `main.py` | Root CLI entry point |
| `agent/config.py` | Load and validate environment configuration |
| `agent/models.py` | Pydantic task, plan, tool call, result, and reflection schemas |
| `agent/task_queue.py` | Read/update local JSON queue and normalize GitHub issue tasks |
| `agent/model_client.py` | `ModelClient` protocol and OpenAI-compatible implementation |
| `agent/prompts.py` | System policy and structured planning/execution prompts |
| `agent/planner.py` | Convert a normalized task to a validated plan |
| `agent/orchestrator.py` | Bounded Plan → tool → verify → correct → result loop |
| `agent/verifier.py` | Execute only allowlisted plan verification commands |
| `agent/reflection.py` | Persist outcome reflections and inert skill/rule proposals |
| `agent/self_eval.py` | Detect agent-source edits and require test/eval success before commit |
| `agent/github_source.py` | Normalize GitHub Issues through the optional `gh` integration |
| `tools/workspace.py` | Workspace-contained file read/write/list operations |
| `tools/command_runner.py` | Allowlisted argv command execution with timeout and captured result |
| `tools/registry.py` | Typed tool registration and validated tool dispatch |
| `tools/git_ops.py` | Dedicated branch, diff, commit, and self-evaluation gates |
| `tools/github_ops.py` | Optional GitHub issue polling and PR creation using `gh` |
| `tasks/tasks.json` | Example local queue |
| `memory/.gitkeep` | Track the directory where runtime reflections are stored |
| `memory/task_state.json` | Runtime-only task completion state, separate from the tracked queue |
| `skills/approved/README.md` | Active-skill policy and approval contract |
| `skills/proposals/.gitkeep` | Location for inert, unapproved generated proposals |
| `evals/cases.json` | Deterministic expected behavior and safety cases |
| `evals/__init__.py` | Make the offline eval runner importable with `python -m evals.run` |
| `evals/run.py` | Offline deterministic evaluation runner |
| `tests/` | Offline unit/integration tests using fakes and temporary directories |
| `README.md` | Setup, configuration, queue schema, execution, safety, and GitHub instructions |

## Task 1: Establish package, configuration, and domain contracts

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `agent/__init__.py`
- Create: `agent/config.py`
- Create: `agent/models.py`
- Create: `tests/test_config.py`
- Create: `tests/test_models.py`

**Interfaces:**
- `Settings.from_env(environ: Mapping[str, str] | None = None) -> Settings`; `TASK_SOURCE` defaults to `local`
- `Task`: `id`, `title`, `description`, `acceptance_criteria`, `source`, `state`
- `Subtask`: `id`, `description`, `files`, `verification`
- `TaskPlan`: `task_id`, ordered `subtasks`, `verification_commands`
- `ToolCall`: `name`, JSON-object `arguments`
- `AgentDecision`: exactly one of `tool_call` or `final_response`
- `TaskResult`: `task_id`, `status: Literal["succeeded", "failed"]`, `summary`, `changed_files`, `error`
- `VerificationResult`: `passed`, `commands`, `failures`
- `Reflection`: `task_id`, `outcome`, `lessons`, `failure_category`, `created_at`

- [ ] **Step 1: Write tests for valid and invalid settings**

```python
def test_settings_read_openai_compatible_endpoint(tmp_path):
    settings = Settings.from_env({
        "MODEL_API_KEY": "test-key",
        "MODEL_BASE_URL": "https://model.example/v1",
        "MODEL_NAME": "test-model",
        "WORKSPACE_ROOT": str(tmp_path),
    })
    assert settings.model_base_url == "https://model.example/v1"
    assert settings.max_steps > 0

def test_settings_reject_nonpositive_step_limit():
    with pytest.raises(SettingsError):
        Settings.from_env({"MAX_STEPS": "0"})
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `python -m pytest tests/test_config.py tests/test_models.py -q`
Expected: FAIL because `agent.config` and `agent.models` do not yet exist.

- [ ] **Step 3: Add package metadata and settings/domain models**

Use `pydantic>=2,<3`, `openai>=1,<3`, `python-dotenv>=1,<2`, and `pytest>=8,<9`; declare pytest under the `dev` optional dependency group. Parse integer limits explicitly, require positive values, resolve workspace paths, and represent allowed commands as JSON arrays of argv prefixes rather than shell strings. Configure positive `MAX_STEPS`, `MAX_CORRECTIONS`, `MAX_RETRIES`, `COMMAND_TIMEOUT_SECONDS`, `POLL_INTERVAL_SECONDS`, and `MAX_POLL_CYCLES`, plus `TASK_SOURCE=local|github` defaulting to `local`. Make `MODEL_API_KEY` required only when constructing the live model client so fake/local tests need no secret. Exclude `.env`, virtual environments, bytecode, `memory/reflection.json`, `memory/task_state.json`, and pytest caches from Git while retaining `memory/.gitkeep`.

- [ ] **Step 4: Run focused tests and verify model invariants**

Run: `python -m pytest tests/test_config.py tests/test_models.py -q`
Expected: PASS, including rejection of malformed task source/state and an `AgentDecision` containing both or neither result alternatives.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore .env.example agent tests
git commit -m "feat: add TuMinh agent contracts and config"
```

## Task 2: Implement local queue, model adapter, and planner

**Files:**
- Create: `agent/task_queue.py`
- Create: `agent/model_client.py`
- Create: `agent/prompts.py`
- Create: `agent/planner.py`
- Create: `tasks/tasks.json`
- Create: `tests/test_task_queue.py`
- Create: `tests/test_planner.py`
- Create: `tests/fakes.py`

**Interfaces:**
- `TaskQueue(queue_path: Path, state_path: Path).next_task() -> Task | None`; never auto-select succeeded or failed tasks
- `TaskQueue(queue_path: Path, state_path: Path).save_result(task_id: str, result: TaskResult) -> None`
- `ModelClient.complete_json(self, system_prompt: str, user_payload: str, response_type: type[TModel]) -> TModel`
- `Planner(model: ModelClient).create_plan(task: Task) -> TaskPlan`
- `tests/fakes.py` defines `ScriptedModel.complete_json`, `FakePlanner.create_plan`, `FakeVerifier.verify`, `FakeReflectionStore.append`, and `FakeToolRegistry.execute`; `ScriptedModel.call_count` increments once per response.

- [ ] **Step 1: Write queue and planner tests**

Test that a task is normalized to `Task`, succeeded and failed outcomes in `memory/task_state.json` prevent re-selection without mutating tracked `tasks/tasks.json`, malformed JSON raises a visible `TaskQueueError`, runtime-state writes are atomic, and a fake model's invalid plan JSON raises a validation error. Assert planner prompts label task descriptions as untrusted data. Create the five fakes listed in the interface block in `tests/fakes.py`; make the scripted model consume responses in order and raise `AssertionError` if a test requests more responses than supplied.

```python
def test_completed_task_is_skipped_without_mutating_queue(tmp_path):
    queue_path = tmp_path / "tasks.json"
    state_path = tmp_path / "task_state.json"
    queue_path.write_text(json.dumps([{
        "id": "one", "title": "Example", "description": "Do one safe change.",
        "acceptance_criteria": ["A focused test passes."],
        "source": "local", "state": "pending",
    }]))
    queue = TaskQueue(queue_path, state_path)
    task = queue.next_task()

    queue.save_result(task.id, TaskResult(
        task_id=task.id, status="succeeded", summary="done",
        changed_files=[], error=None,
    ))
    assert queue.next_task() is None
    assert json.loads(queue_path.read_text())[0]["state"] == "pending"
```

- [ ] **Step 2: Run focused tests and confirm they fail**

Run: `python -m pytest tests/test_task_queue.py tests/test_planner.py -q`
Expected: FAIL because the queue, model protocol, and planner modules do not exist.

- [ ] **Step 3: Implement normalized task loading and planning**

Validate every task with `Task.model_validate`; keep the sample queue immutable and write succeeded/failed state to the ignored `memory/task_state.json` through a temporary file followed by atomic replacement. Do not automatically retry a failed queue item; document resetting its runtime state to requeue it. Define `TModel = TypeVar("TModel", bound=BaseModel)` and a `Protocol` method `complete_json(self, system_prompt: str, user_payload: str, response_type: type[TModel]) -> TModel`; implement the live adapter with `OpenAI(base_url=settings.model_base_url, api_key=settings.model_api_key)`, Chat Completions JSON output, and `response_type.model_validate_json`. Use explicit system policy separate from untrusted task payload.

- [ ] **Step 4: Run focused tests, including malformed model output**

Run: `python -m pytest tests/test_task_queue.py tests/test_planner.py -q`
Expected: PASS; malformed queue/model output must raise typed errors rather than produce an empty task or plan.

- [ ] **Step 5: Commit**

```bash
git add agent tasks tests
git commit -m "feat: add task queue and model planning"
```

## Task 3: Add contained workspace and allowlisted command tools

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/workspace.py`
- Create: `tools/command_runner.py`
- Create: `tools/registry.py`
- Create: `tests/test_workspace.py`
- Create: `tests/test_command_runner.py`
- Create: `tests/test_registry.py`

**Interfaces:**
- `Workspace(root: Path).read_file(path: str) -> str`
- `Workspace(root: Path).write_file(path: str, content: str) -> None`
- `Workspace(root: Path).list_files(path: str = ".") -> list[str]`
- `CommandRunner(allowed_argv_prefixes: Sequence[Sequence[str]], timeout_seconds: int).run(argv: Sequence[str], cwd: Path) -> CommandResult`
- `CommandResult`: `argv`, `exit_code`, `stdout`, `stderr`, and `timed_out`
- `ToolRegistry.register(name: str, schema: type[BaseModel], handler: Callable) -> None`
- `ToolRegistry.execute(call: ToolCall) -> dict[str, object]`
- `tests/fakes.py` defines `FakeCommandRunner(exit_code: int = 0)` with `run(argv: Sequence[str], cwd: Path) -> CommandResult` and a `calls` list of argv vectors.
- Raise `WorkspaceError` for path violations and `CommandError` for rejected/failed command execution.

- [ ] **Step 1: Write path-boundary, timeout, allowlist, and registry tests**

Cover `..` traversal, absolute paths outside root, symlink escape, allowed argv prefix, rejected executable/arguments, timeout, nonzero exit status, unknown tool, invalid arguments, and handler exceptions being surfaced.

```python
def test_workspace_rejects_parent_traversal(tmp_path):
    workspace = Workspace(tmp_path)
    with pytest.raises(WorkspaceError):
        workspace.write_file("..\\escape.txt", "no")
    assert not (tmp_path.parent / "escape.txt").exists()

def test_command_runner_rejects_unlisted_executable(tmp_path):
    runner = CommandRunner([[sys.executable, "-c"]], timeout_seconds=2)
    with pytest.raises(CommandError):
        runner.run(["git", "status"], cwd=tmp_path)
```

- [ ] **Step 2: Run focused tests and confirm they fail**

Run: `python -m pytest tests/test_workspace.py tests/test_command_runner.py tests/test_registry.py -q`
Expected: FAIL because the tool modules do not exist.

- [ ] **Step 3: Implement tools without shell execution**

Resolve target paths and verify containment after resolving symlinks. Use `subprocess.run(argv, shell=False, cwd=cwd, timeout=self.timeout_seconds)`; reject empty argv, reject unconfigured argv prefixes, and return stdout/stderr/exit code in a typed result. Validate registry arguments with the declared Pydantic schema before calling the handler.

- [ ] **Step 4: Run focused tests and verify all escape attempts fail closed**

Run: `python -m pytest tests/test_workspace.py tests/test_command_runner.py tests/test_registry.py -q`
Expected: PASS; an escaped path or unapproved command raises a typed tool error and does not mutate files or spawn the command.

- [ ] **Step 5: Commit**

```bash
git add tools tests
git commit -m "feat: add safe workspace and command tools"
```

## Task 4: Implement the bounded autonomous execution loop

**Files:**
- Create: `agent/orchestrator.py`
- Create: `agent/verifier.py`
- Create: `tests/test_orchestrator.py`
- Create: `tests/test_verifier.py`
- Modify: `tests/fakes.py`

**Interfaces:**
- `Verifier` protocol: `verify(plan: TaskPlan, changed_files: Sequence[str]) -> VerificationResult`
- `Orchestrator(planner: Planner, model: ModelClient, tools: ToolRegistry, settings: Settings, verifier: Verifier, reflections: ReflectionStore).run(task: Task) -> TaskResult`
- `CommandVerifier(command_runner: CommandRunner).verify(plan: TaskPlan, changed_files: Sequence[str]) -> VerificationResult`
- Each model action is an `AgentDecision`; each tool request passes through `ToolRegistry.execute`.

- [ ] **Step 1: Write loop tests with a fake planner, scripted model, fake tools, and verifier**

Test `CommandVerifier` invokes only plan verification argv accepted by `CommandRunner`, returns a failed result for nonzero exit, timeout, and no configured verification command. Test orchestrator order is plan → validated tool call → tool result → verification → successful result. Add cases for max-step exhaustion, malformed action, failing verification followed by one bounded correction and pass, repeated failed verification, and tool exception. Assert failure results never have success status. Define `build_orchestrator(max_steps: int, decisions: Sequence[AgentDecision]) -> tuple[Orchestrator, ScriptedModel]` in this test file; inject a fixed valid plan, `FakeVerifier`, `FakeReflectionStore`, and `FakeToolRegistry`. Define `sample_task() -> Task` in `tests/fakes.py` with one local task and an explicit acceptance criterion.

```python
def test_command_verifier_fails_on_nonzero_command():
    verifier = CommandVerifier(command_runner=FakeCommandRunner(exit_code=3))
    result = verifier.verify(
        TaskPlan(task_id="t1", subtasks=[], verification_commands=[["pytest", "-q"]]),
        changed_files=[],
    )
    assert result.passed is False
    assert result.failures

def test_orchestrator_stops_at_max_steps():
    decisions = [
        AgentDecision(tool_call=ToolCall(name="read_file", arguments={"path": "README.md"}))
        for _ in range(3)
    ]
    agent, model = build_orchestrator(max_steps=2, decisions=decisions)
    result = agent.run(sample_task())
    assert result.status == "failed"
    assert model.call_count == 2
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `python -m pytest tests/test_verifier.py tests/test_orchestrator.py -q`
Expected: FAIL because `agent.verifier` and `agent.orchestrator` do not exist.

- [ ] **Step 3: Implement bounded iteration and error propagation**

Define a `Verifier` protocol and implement `CommandVerifier` to execute each `TaskPlan.verification_commands` argv through the allowlisted runner and collect exit status, timeout, and diagnostic output into `VerificationResult`. Stop the orchestrator after `Settings.max_steps`; use `Settings.max_corrections` for verification repair attempts. Validate every model response before dispatch. Call reflection persistence on both success and failure; if persistence fails, surface that error and do not report a successful completed task. Retry only configured transient model errors, with a strict retry count.

- [ ] **Step 4: Run loop tests and verify exact step limits**

Run: `python -m pytest tests/test_verifier.py tests/test_orchestrator.py -q`
Expected: PASS; unapproved verification argv is rejected and a script with more actions than `max_steps` stops at the limit with a failed result.

- [ ] **Step 5: Commit**

```bash
git add agent/verifier.py agent/orchestrator.py tests
git commit -m "feat: add bounded agent execution loop"
```

## Task 5: Add reflection storage and gated skill proposals

**Files:**
- Create: `agent/reflection.py`
- Create: `memory/.gitkeep`
- Create: `skills/approved/README.md`
- Create: `skills/proposals/.gitkeep`
- Create: `tests/test_reflection.py`

**Interfaces:**
- `HumanApproval(approved_by: str, confirmed_at: datetime)`
- `ReflectionStore(path: Path, secret_values: Sequence[str] = ())`
- `ReflectionStore.append(reflection: Reflection) -> None`
- `ReflectionStore.read_all() -> list[Reflection]`
- `ReflectionStore.propose_skill(slug: str, content: str) -> Path`
- `ReflectionStore.activate_proposal(path: Path, *, eval_result: VerificationResult, approval: HumanApproval) -> Path`
- `tests/fakes.py` defines `FakeSelfEvaluator(passed: bool).verify_before_commit(changed_files: Sequence[str]) -> VerificationResult`.

- [ ] **Step 1: Write persistence, redaction, proposal, and activation-gate tests**

Assert append/read round-trips multiple reflections; configured secrets and full command output are omitted; malformed existing JSON raises `ReflectionError`; generated proposals remain under `skills/proposals`; activation rejects a failed eval result or absent approval object and only copies a valid proposal after both gates pass.

```python
def test_failed_eval_does_not_activate_proposal(tmp_path):
    store = ReflectionStore(tmp_path / "memory" / "reflection.json")
    proposal = store.propose_skill("safe-test", "Prefer focused regression tests.")
    approval = HumanApproval(approved_by="operator", confirmed_at=datetime.now(UTC))
    failed_eval = VerificationResult(passed=False, commands=[], failures=["eval failed"])

    with pytest.raises(ReflectionError):
        store.activate_proposal(proposal, eval_result=failed_eval, approval=approval)
    assert not (tmp_path / "skills" / "approved" / "safe-test.md").exists()
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `python -m pytest tests/test_reflection.py -q`
Expected: FAIL because the reflection store does not exist.

- [ ] **Step 3: Implement validated append-only reflections and inert proposals**

Write reflections to `memory/reflection.json` through atomic replacement and validate every stored item; create the file on first append. Sanitize constructor-supplied secret values before persistence. Restrict slugs to safe filenames, reject proposal paths outside the proposals directory, and copy approved content into the active directory only through `activate_proposal` after both gates pass. Do not expose proposal activation as an agent tool; require the separate human-facing CLI command in Task 8.

- [ ] **Step 4: Run focused tests, including activation denials**

Run: `python -m pytest tests/test_reflection.py -q`
Expected: PASS; either missing approval gate raises without changing active skills.

- [ ] **Step 5: Commit**

```bash
git add agent/reflection.py memory skills tests
git commit -m "feat: add reflection and skill proposal gates"
```

## Task 6: Add deterministic evals and self-modification test gate

**Files:**
- Create: `agent/self_eval.py`
- Create: `evals/__init__.py`
- Create: `evals/cases.json`
- Create: `evals/run.py`
- Create: `tests/test_self_eval.py`
- Create: `tests/test_evals.py`

**Interfaces:**
- `SelfEvaluator(agent_root: Path, command_runner: CommandRunner).verify_before_commit(changed_files: Sequence[str]) -> VerificationResult`
- `run_evals() -> int`
- `evaluate_proposal(path: Path) -> VerificationResult`
- Evaluation cases use validated fixtures and fake model/tool responses; no network calls.

- [ ] **Step 1: Write tests for agent-change detection and eval failure**

Cover changes inside `agent/`, `tools/`, `tests/`, `evals/`, or `skills/approved/`; no self-test for unrelated product-only edits; require both pytest and deterministic eval success for agent edits; failure blocks the commit gate. Add eval cases for invalid tool arguments, untrusted task instructions, explicit failure reporting, and inert skill activation. Verify `evaluate_proposal` applies the deterministic cases to the candidate file and returns a failed result for an invalid or unsafe proposal.

```python
def test_agent_change_requires_tests_and_evals(tmp_path):
    evaluator = SelfEvaluator(tmp_path, command_runner=FakeCommandRunner(exit_code=1))
    result = evaluator.verify_before_commit(["agent/orchestrator.py"])
    assert result.passed is False

def test_product_only_change_skips_agent_eval(tmp_path):
    evaluator = SelfEvaluator(tmp_path, command_runner=FakeCommandRunner(exit_code=0))
    result = evaluator.verify_before_commit(["src/product.py"])
    assert result.passed is True
    assert evaluator.command_runner.calls == []
```

- [ ] **Step 2: Run focused tests and confirm they fail**

Run: `python -m pytest tests/test_self_eval.py tests/test_evals.py -q`
Expected: FAIL because self-evaluation and eval runner are missing.

- [ ] **Step 3: Implement deterministic offline evals and self-test policy**

Run test/eval commands using the allowlisted command runner and the current agent workspace. Detect changed paths from the Git change set. Return a failed `VerificationResult` on any nonzero result; never catch and convert it to success. Ensure eval fixtures contain no credentials or network dependency. `evaluate_proposal` validates the candidate and runs deterministic proposal-specific fixture checks before returning a passed result.

- [ ] **Step 4: Run unit tests and eval runner**

Run: `python -m pytest tests/test_self_eval.py tests/test_evals.py -q`
Run: `python -m evals.run`
Expected: Both commands succeed offline; a deliberately failing fake eval blocks the commit gate.

- [ ] **Step 5: Commit**

```bash
git add agent/self_eval.py evals tests
git commit -m "test: add offline agent self-evaluation"
```

## Task 7: Implement dedicated-branch Git operations and commit gates

**Files:**
- Create: `tools/git_ops.py`
- Create: `tests/test_git_ops.py`
- Modify: `agent/orchestrator.py`

**Interfaces:**
- `GitOperations.ensure_task_branch(task_id: str) -> str`
- `GitOperations.changed_files() -> list[str]`
- `GitOperations.commit_verified(task_id: str, message: str, *, task_verification: VerificationResult, self_evaluator: SelfEvaluator) -> str`
- `GitOperations(workspace: Path, command_runner: CommandRunner)`
- The orchestrator may commit only after task verification succeeds.

- [ ] **Step 1: Write Git guard tests using an injected fake command runner**

Assert branch name is derived from a sanitized task id, current `main`/base branch is never modified, commit is refused after failed task verification or self-eval, commit is refused when the change set is ambiguous, and commit uses only the configured Git argv commands.

```python
def test_failed_task_verification_never_commits(tmp_path):
    runner = FakeCommandRunner(exit_code=0)
    git = GitOperations(workspace=tmp_path, command_runner=runner)
    failed = VerificationResult(passed=False, commands=["pytest"], failures=["failed"])
    evaluator = FakeSelfEvaluator(passed=True)

    with pytest.raises(GitOperationError):
        git.commit_verified(
            "issue-4", "feat: change", task_verification=failed,
            self_evaluator=evaluator,
        )
    assert not any(argv[0] == "git" and argv[1] == "commit" for argv in runner.calls)
```

- [ ] **Step 2: Run focused tests and confirm they fail**

Run: `python -m pytest tests/test_git_ops.py -q`
Expected: FAIL because `tools.git_ops` does not exist.

- [ ] **Step 3: Implement branch and verified-commit flow**

Create/switch to a task branch before file-writing tools are enabled. Inspect branch/status/diff using fixed Git argv calls. Before a commit, require successful task verification and, if agent files changed, `SelfEvaluator.verify_before_commit` success. Do not use `git add .`; stage only validated paths from the task's change set. Refuse to commit pre-existing unrelated changes.

- [ ] **Step 4: Run Git tests and verify failure never stages/commits**

Run: `python -m pytest tests/test_git_ops.py tests/test_orchestrator.py -q`
Expected: PASS; fake Git history contains no commit for each failed gate.

- [ ] **Step 5: Commit**

```bash
git add tools/git_ops.py agent/orchestrator.py tests
git commit -m "feat: gate commits on branch and verification"
```

## Task 8: Add optional GitHub Issues/PR adapter and runnable CLI

**Files:**
- Create: `agent/github_source.py`
- Create: `tools/github_ops.py`
- Create: `main.py`
- Create: `tests/test_github_source.py`
- Create: `tests/test_cli.py`
- Modify: `agent/task_queue.py`
- Modify: `agent/config.py`

**Interfaces:**
- `GitHubIssueSource.next_task() -> Task | None`; use normalized ID `github:<number>` and the same runtime completion state to avoid replaying processed issues
- `GitHubOperations.create_pull_request(branch: str, title: str, body: str) -> str`
- `main(argv: Sequence[str] | None = None) -> int`
- CLI flags: `--once` (default) and `--poll`; polling interval must be positive and bounded.
- Add an explicit `approve-skill-proposal` subcommand that displays the proposal and prompts before creating `HumanApproval`; no agent tool can call this flow.
- `request_skill_approval(input_fn: Callable[[str], str] = input) -> HumanApproval | None` returns an approval only for an explicit affirmative response and records `getpass.getuser()` plus a UTC timestamp.

- [ ] **Step 1: Write fake-`gh` tests and CLI tests**

Test issue JSON normalization, missing `gh`/authentication error reporting, PR creation only after successful verification and commit, default one-task behavior, and polling stop/interval/`MAX_POLL_CYCLES` limits. Test local queue is selected by default and GitHub source requires explicit `TASK_SOURCE=github`; processed issue IDs must be skipped. Test the skill-approval subcommand rejects a negative response and only activates a proposal after a positive response plus successful eval. Assert queue mode works without GitHub environment values.

```python
def test_skill_proposal_requires_explicit_cli_confirmation(tmp_path):
    approval = request_skill_approval(input_fn=lambda _: "no")
    assert approval is None
```

- [ ] **Step 2: Run focused tests and confirm they fail**

Run: `python -m pytest tests/test_github_source.py tests/test_cli.py -q`
Expected: FAIL because GitHub adapter and CLI are missing.

- [ ] **Step 3: Implement optional GitHub and CLI wiring**

Call `load_dotenv()` before `Settings.from_env`. Select local queue by default and GitHub only when `TASK_SOURCE=github`. Use `["gh", "issue", "list", "--json", "number,title,body,labels,state"]` and `["gh", "pr", "create", "--title", title, "--body", body, "--head", branch]` only via fixed argv commands through the command runner. Parse JSON and validate issue data as untrusted task fields. `--once` returns nonzero for a failed task; `--poll` exits on fatal configuration/tool errors, stops at `MAX_POLL_CYCLES`, and reports each task result. The separate approval subcommand runs `evaluate_proposal(path)`, displays the candidate and eval result, reads an explicit affirmative response, creates `HumanApproval`, and only then calls `activate_proposal`.

- [ ] **Step 4: Run queue-only and fake-GitHub tests**

Run: `python -m pytest tests/test_github_source.py tests/test_cli.py -q`
Expected: PASS without network access or credentials.

- [ ] **Step 5: Commit**

```bash
git add main.py agent tools tests
git commit -m "feat: add CLI and optional GitHub integration"
```

## Task 9: Complete documentation and end-to-end offline verification

**Files:**
- Create: `README.md`
- Modify: `.env.example`
- Modify: `tasks/tasks.json`
- Modify: `pyproject.toml`
- Create: `tests/test_starter_flow.py`

**Interfaces:**
- Document `python -m venv .venv`, dependency installation, `python main.py --once`, optional `python main.py --poll`, local task schema, model configuration, GitHub setup, and the safety boundary.
- The sample queue must use a harmless task with an explicit acceptance criterion and an allowlisted test command.

- [ ] **Step 1: Write an offline end-to-end starter-flow test**

Use a temporary workspace, `TaskQueue`, fake model, fake registered tools, fake verifier, and temporary `ReflectionStore`. Assert a queued task reaches verified completion, queue state is updated, reflection is persisted, no live API is constructed, and no PR/commit is attempted by the test fixture.

```python
def test_starter_flow_persists_success_without_external_services(tmp_path):
    app = make_test_app(tmp_path, model=ScriptedModel([AgentDecision(final_response="done")]))
    result = app.run_once()

    assert result.status == "succeeded"
    assert json.loads((tmp_path / "memory" / "task_state.json").read_text())
    assert ReflectionStore(tmp_path / "memory" / "reflection.json").read_all()
    assert app.github.calls == []
```

Define `TestApp.run_once() -> TaskResult` and `make_test_app(tmp_path: Path, model: ScriptedModel) -> TestApp` in this test module. The factory injects a fixed planner, a passing fake verifier, temporary queue/state/reflection paths, and a GitHub fake with an initially empty `calls` list.

- [ ] **Step 2: Run the end-to-end test and confirm it fails**

Run: `python -m pytest tests/test_starter_flow.py -q`
Expected: FAIL until CLI/application wiring is complete.

- [ ] **Step 3: Write setup and operation documentation**

Give copyable PowerShell and POSIX setup commands, explain each required `.env` variable including `TASK_SOURCE`, show a valid queue item, distinguish local mode from optional GitHub mode, state that the tool restrictions are not an OS sandbox, explain failed-task reset via `memory/task_state.json`, and explain how proposals are reviewed/activated. Include expected CLI outcomes for success and failure.

- [ ] **Step 4: Run offline verification after dependencies are installed**

Run: `python -m pip install -e ".[dev]"`
Run: `python -m pytest -q`
Run: `python -m evals.run`
Run: `python main.py --help`
Expected: all commands succeed without model/GitHub credentials or network calls after dependency installation.

- [ ] **Step 5: Commit**

```bash
git add README.md .env.example tasks pyproject.toml tests
git commit -m "docs: document TuMinh agent setup and operation"
```

## Final acceptance check

- [ ] A local queue task runs without GitHub credentials.
- [ ] Unit/integration tests and deterministic evals run without network access.
- [ ] Malformed model responses, queue entries, tool arguments, and failed commands are reported explicitly.
- [ ] File paths remain within the workspace and subprocess calls use `shell=False`, argv allowlisting, and timeouts.
- [ ] A task failure or agent self-test failure blocks commit and PR creation.
- [ ] Agent changes are made on a task branch; no operation merges automatically.
- [ ] Skill/rule proposals remain inactive until eval passes and a human approves.
