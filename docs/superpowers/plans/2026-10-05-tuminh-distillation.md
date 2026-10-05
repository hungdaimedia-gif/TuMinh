# TuMinh Distillation and Model Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional distillation subsystem that imports authorized observable trajectories, curates a verified dataset, evaluates an optional local model candidate, and supports human-gated version promotion and rollback.

**Architecture:** Implement `distill/` as a separate CPU-testable package that consumes the runtime JSONL contract and manually supplied authorized exports. Keep CUDA training and GGUF/Ollama operations behind optional adapters; make promotion an atomic registry-pointer change after non-regression/safety evals and explicit human approval.

**Tech Stack:** Python 3.11+, Pydantic 2, stdlib JSONL/TOML/hashlib, optional Linux/WSL2 training environment with Unsloth/PyTorch/CUDA, optional GGUF/Ollama tooling, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-tuminh-distillation-engine-design.md`

**Prerequisite:** Complete `docs/superpowers/plans/2026-10-05-tuminh-autonomous-agent.md` through Task 10 so the sanitized runtime journal and resolver interface exist.

## Global Constraints

- Accept TuMinh runtime exports and manually supplied external exports only when the user declares that they have the right to use them.
- Store task, plan, emitted responses/actions, tool I/O, diffs, verification results, and concise reflections. Never ingest hidden chain-of-thought, provider system prompts, credentials, or environment values.
- Only successful trajectories with passing verification/evals can enter the curated training set.
- Make local QLoRA training optional and configurable. RTX 3060 Ti 8GB is an optional profile, not a promise of fixed VRAM usage or throughput.
- Require candidate pass rate at least equal to baseline and 100% pass on safety-critical evals before it is eligible for promotion.
- Require an explicit human confirmation to promote. Keep versioned model tags and a tested rollback operation.
- Keep raw exports, curated datasets, checkpoints, adapters, and model weights local and out of Git.
- External exports are manually supplied files, not scraped or harvested from provider sessions.
- The runtime never uploads trajectories or turns them into training data by itself.
- A candidate is eligible only when its aggregate pass rate is at least the active model's pass rate on the same eval set, every safety-critical eval passes, and required provenance/benchmark metadata are present.
- Normal CI is CPU-only; real CUDA training and Ollama integration require explicit local opt-in.

---

## File map

| File | Responsibility |
|---|---|
| `distill/schema.py` | Pydantic schemas for trajectories, provenance, dataset rows, eval reports, approvals, and candidate manifests |
| `distill/importers.py` | Explicit runtime export and manual external-export import |
| `distill/curator.py` | Secret redaction, successful-task eligibility, normalization, dataset manifest and atomic JSONL output |
| `distill/benchmark.py` | Run the same pinned model eval suite and compute promotion eligibility |
| `distill/registry.py` | Versioned candidate/active/previous pointers, human approval, promotion and rollback |
| `distill/trainer.py` | Optional QLoRA training adapter and measured hardware profile |
| `distill/deploy.py` | Optional GGUF conversion and versioned Ollama candidate installation |
| `distill/cli.py` | Explicit import/export/curate/train/benchmark/promote/rollback/purge commands |
| `distill/config.example.toml` | Model, license, training, hardware-profile, and eval settings |
| `distill/requirements-training.txt` | Separate training environment dependencies; never a base runtime dependency |
| `distill/data/raw/` | Local imported exports; ignored by Git |
| `distill/data/curated/` | Local curated datasets/manifests; ignored by Git |
| `distill/models/registry.json` | Local active/candidate/previous model pointers; ignored by Git |
| `distill/models/candidates/` | Local adapters, GGUFs, and checkpoints; ignored by Git |
| `distill/evals/cases.json` | Synthetic, versioned model-eval cases with safety-critical labels |
| `tests/test_distill_schema.py` | Schema and forbidden-field tests |
| `tests/test_distill_importers.py` | Runtime/manual import and provenance tests |
| `tests/test_distill_curator.py` | Redaction, eligibility, manifest, and atomic-output tests |
| `tests/test_distill_benchmark.py` | Same-suite score comparison and safety threshold tests |
| `tests/test_distill_registry.py` | Approval, promotion, atomic pointer, and rollback tests |
| `tests/test_distill_trainer.py` | CPU config validation and fake trainer tests |
| `tests/test_distill_deploy.py` | Fake GGUF/Ollama operations and failure preservation tests |
| `tests/test_distill_cli.py` | CLI safety and no-network behavior |
| `tests/distill_fakes.py` | Synthetic trajectory/model manifests and deterministic adapters for CPU tests |
| `.gitignore` | Exclude local source exports, datasets, registries, checkpoints, adapters, and model files |
| `README.md` | Explain data provenance, CPU workflow, optional training environment, local model setup, approval, and rollback |
| `agent/model_client.py` | Resolve `MODEL_NAME=tuminh-active` using the injected model-name resolver |

## Task 1: Define trajectory, provenance, dataset, and candidate schemas

**Files:**
- Create: `distill/__init__.py`
- Create: `distill/schema.py`
- Modify: `pyproject.toml`
- Create: `tests/test_distill_schema.py`
- Create: `tests/distill_fakes.py`
- Create: `tests/fixtures/distill/valid-runtime-export.jsonl`
- Create: `tests/fixtures/distill/authorized-external-export.jsonl`
- Create: `tests/fixtures/distill/invalid-private-fields.jsonl`

**Interfaces:**
- `TrajectoryExport`: schema version, trajectory ID, source type, provenance, task outcome, verification/eval results, ordered runtime events
- `ImportProvenance`: `source_name`, `license_or_authorization`, `authorized_by`, `imported_at`, `source_sha256`
- `TrainingMessage`: role is exactly `user`, `assistant`, or `tool`; includes string content and optional tool name
- `TrainingRow`: task ID, ordered messages, verification/eval evidence, provenance reference
- `DatasetManifest`: dataset version, source hashes, curation version, redaction counts, created time, row count
- `CandidateManifest`: model ID/revision, base model/license, dataset hash, training config, runtime/hardware metadata, benchmark ID
- `tests/distill_fakes.py` defines `valid_trajectory(outcome: Literal["succeeded", "failed"] = "succeeded", verification_passed: bool = True, eval_passed: bool = True) -> TrajectoryExport` using synthetic fields only.
- The `valid-runtime-export.jsonl` fixture contains a schema-version header followed by one task event and one successful verification event. The external fixture contains only user/assistant/tool messages with synthetic content; the invalid fixture differs by containing a prohibited nested `thinking` field.

- [ ] **Step 1: Write schema acceptance and forbidden-field tests**

Test valid user/assistant/tool rows, invalid roles such as `thought` and `system`, missing authorization, invalid task outcomes, duplicate event sequence numbers, and recursively nested `thought`, `thinking`, `analysis`, system-prompt, developer-prompt, credential, or environment fields.

```python
def test_training_row_rejects_private_reasoning_role():
    with pytest.raises(ValidationError):
        TrainingRow.model_validate({
            "task_id": "task-1",
            "messages": [{"role": "thought", "content": "private reasoning"}],
            "verification_passed": True,
            "eval_passed": True,
            "provenance_id": "source-1",
        })
```

- [ ] **Step 2: Run schema tests and verify the expected failure**

Run: `python -m pytest tests/test_distill_schema.py -q`
Expected: FAIL because `distill.schema` does not exist.

- [ ] **Step 3: Implement strict Pydantic schemas**

Set `extra="forbid"` on every external-facing model. Validate all IDs, source types, outcomes, event ordering, metadata, and message roles. Recursively reject prohibited keys before any normalizer can discard them. Require a nonempty license/authorization declaration for external provenance.

- [ ] **Step 4: Run schema tests**

Run: `python -m pytest tests/test_distill_schema.py -q`
Expected: PASS; each prohibited field and role is rejected and valid synthetic fixtures round-trip.

- [ ] **Step 5: Commit**

```bash
git add distill/__init__.py distill/schema.py pyproject.toml tests/test_distill_schema.py tests/distill_fakes.py tests/fixtures/distill
git commit -m "feat: define distillation data contracts"
```

## Task 2: Add explicit runtime export and authorized external import

**Files:**
- Create: `distill/importers.py`
- Create: `distill/cli.py`
- Create: `tests/test_distill_importers.py`

**Interfaces:**
- `RuntimeExporter(journal_root: Path).export(task_id: str, output_path: Path) -> Path`
- `ExternalImporter.import_file(path: Path, *, source_name: str, license_or_authorization: str, authorized_by: str) -> list[TrajectoryExport]`
- `DistillImportError` is a typed, actionable importer error; malformed input is never returned as an empty successful list.

- [ ] **Step 1: Write explicit-export and provenance tests**

Test that runtime export copies only one validated task file to the requested local path, rejects path traversal task IDs, and does not call any network API. Test external import requires source/license/operator metadata, records a SHA-256 source hash, and rejects private reasoning keys and malformed JSONL without writing partial output.

```python
def test_external_import_requires_authorization(tmp_path):
    source = tmp_path / "export.jsonl"
    fixture = Path(__file__).parent / "fixtures" / "distill" / "authorized-external-export.jsonl"
    source.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(DistillImportError):
        ExternalImporter().import_file(
            source, source_name="provider-export",
            license_or_authorization="", authorized_by="operator",
        )
```

- [ ] **Step 2: Run importer tests and verify the expected failure**

Run: `python -m pytest tests/test_distill_importers.py -q`
Expected: FAIL because `distill.importers` does not exist.

- [ ] **Step 3: Implement import as an explicit local file operation**

Implement runtime export by reading the per-task JSONL journal and writing validated JSONL via a temporary file and atomic replace. Implement external import only from the caller-supplied path; require nonblank authorization metadata, hash the original bytes, map only observable user/assistant/tool records, and reject all forbidden or unknown reasoning keys before mapping.

- [ ] **Step 4: Run importer tests**

Run: `python -m pytest tests/test_distill_importers.py -q`
Expected: PASS with no network calls, no provider scraping, and no partial output after invalid input.

- [ ] **Step 5: Commit**

```bash
git add distill/importers.py distill/cli.py tests/test_distill_importers.py
git commit -m "feat: add explicit trajectory import and export"
```

## Task 3: Curate eligible trajectories into a redacted dataset

**Files:**
- Create: `distill/curator.py`
- Create: `tests/test_distill_curator.py`
- Modify: `.gitignore`

**Interfaces:**
- `Curator(secret_values: Sequence[str]).curate(records: Sequence[TrajectoryExport], output_path: Path) -> DatasetManifest`
- `Curator.purge(source_sha256: str, raw_root: Path, curated_root: Path) -> PurgeResult`

- [ ] **Step 1: Write eligibility, redaction, and atomic-output tests**

Test that failed tasks or tasks missing either verification/eval pass are excluded; successful rows redact configured secrets in nested values; manifests contain source hashes and authorization metadata; a malformed record prevents the complete output file from being replaced; and purge removes only records matching the requested source hash.

```python
def test_curator_excludes_failed_task(tmp_path):
    record = valid_trajectory(outcome="failed", verification_passed=False, eval_passed=False)
    output = tmp_path / "curated.jsonl"

    manifest = Curator(secret_values=[]).curate([record], output)

    assert manifest.row_count == 0
    assert output.read_text(encoding="utf-8") == ""
```

- [ ] **Step 2: Run curator tests and verify the expected failure**

Run: `python -m pytest tests/test_distill_curator.py -q`
Expected: FAIL because `distill.curator` does not exist.

- [ ] **Step 3: Implement deterministic all-or-nothing curation**

Validate the whole input batch before opening the destination. Include only succeeded records with passing task verification and required eval evidence. Convert allowed observable events to `user`, `assistant`, and `tool` messages; recursively redact configured secret values; calculate source hashes and redaction counts; write dataset and manifest using temporary files and atomic replacement. Add ignores for raw/curated data and keep the purge operation scoped to exact source hashes.

- [ ] **Step 4: Run curator tests**

Run: `python -m pytest tests/test_distill_curator.py -q`
Expected: PASS; failed/unverified inputs produce no training rows and invalid batches preserve the previous dataset.

- [ ] **Step 5: Commit**

```bash
git add distill/curator.py tests/test_distill_curator.py .gitignore
git commit -m "feat: curate redacted successful trajectories"
```

## Task 4: Benchmark baseline and candidate on the same eval set

**Files:**
- Create: `distill/benchmark.py`
- Create: `distill/evals/cases.json`
- Create: `tests/test_distill_benchmark.py`
- Modify: `tests/distill_fakes.py`

**Interfaces:**
- `ModelEvaluator.evaluate(model_name: str, cases: Sequence[EvalCase]) -> ModelScore`
- `BenchmarkRunner(evaluator: ModelEvaluator, eval_set: Sequence[EvalCase]).compare(baseline: str, candidate: str, metadata: BenchmarkMetadata) -> BenchmarkReport`
- `is_candidate_eligible(baseline_pass_rate: float, candidate_pass_rate: float, safety_results: Mapping[str, bool], metadata: BenchmarkMetadata) -> bool`
- `BenchmarkReport.eligible_for_promotion -> bool`
- `tests/distill_fakes.py` defines `complete_benchmark_metadata() -> BenchmarkMetadata` with synthetic model, dataset, eval, software, and hardware identifiers.

- [ ] **Step 1: Write exact threshold tests with fake evaluator scores**

Cover candidate score equal to baseline (eligible if all safety cases pass), lower aggregate score (ineligible), one failed safety-critical case (ineligible regardless of aggregate score), and missing model/dataset/eval metadata (ineligible).

```python
def test_any_failed_safety_case_blocks_candidate():
    eligible = is_candidate_eligible(
        baseline_pass_rate=0.80,
        candidate_pass_rate=0.95,
        safety_results={"tool-boundary": False},
        metadata=complete_benchmark_metadata(),
    )
    assert eligible is False
```

- [ ] **Step 2: Run benchmark tests and verify the expected failure**

Run: `python -m pytest tests/test_distill_benchmark.py -q`
Expected: FAIL because `distill.benchmark` does not exist.

- [ ] **Step 3: Implement reproducible score comparison**

Run baseline and candidate against the same pinned eval set, tool limits, and decoding configuration. Store model IDs/revisions, dataset hash, eval-set version, pass counts/rate, safety-critical results, package/hardware metadata, and resource measurements. Treat missing data, evaluator errors, and failed critical cases as ineligible; do not convert them to a passing score.

- [ ] **Step 4: Run benchmark tests**

Run: `python -m pytest tests/test_distill_benchmark.py -q`
Expected: PASS; only non-regressing candidates with every safety case passing are eligible.

- [ ] **Step 5: Commit**

```bash
git add distill/benchmark.py distill/evals/cases.json tests/test_distill_benchmark.py
git commit -m "test: add candidate model benchmark gates"
```

## Task 5: Add local model registry and human-gated promotion/rollback

**Files:**
- Create: `distill/registry.py`
- Create: `tests/test_distill_registry.py`
- Modify: `tests/distill_fakes.py`

**Interfaces:**
- `HumanApproval(approved_by: str, confirmed_at: datetime)`
- `DistillModelRegistry(path: Path).register_candidate(manifest: CandidateManifest) -> None`
- `DistillModelRegistry.initialize_active(model_tag: str) -> None` is available only for explicit first-time local setup.
- `DistillModelRegistry.read() -> RegistryState`
- `DistillModelRegistry.promote(candidate_id: str, report: BenchmarkReport, approval: HumanApproval) -> RegistryState`
- `DistillModelRegistry.rollback() -> RegistryState`
- `DistillModelRegistry.resolve_active_model_name(alias: str) -> str`
- `tests/distill_fakes.py` defines `valid_candidate_manifest(candidate_id: str) -> CandidateManifest`, `ineligible_benchmark_report() -> BenchmarkReport`, and `now_utc() -> datetime`.

- [ ] **Step 1: Write registry, approval, atomicity, and rollback tests**

Assert failed/incomplete reports cannot promote, missing approval cannot promote, a valid approval updates active and preserves the old version as previous, rollback restores the previous version, and interrupted/invalid registry writes leave the last valid pointer readable.

```python
def test_promotion_requires_benchmark_eligibility_and_approval(tmp_path):
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active("baseline-v1")
    registry.register_candidate(valid_candidate_manifest("candidate-v2"))
    with pytest.raises(PromotionError):
        registry.promote(
            "candidate-v2",
            report=ineligible_benchmark_report(),
            approval=HumanApproval(approved_by="operator", confirmed_at=now_utc()),
        )
    assert registry.read().active_model == "baseline-v1"
```

- [ ] **Step 2: Run registry tests and verify the expected failure**

Run: `python -m pytest tests/test_distill_registry.py -q`
Expected: FAIL because `distill.registry` does not exist.

- [ ] **Step 3: Implement atomic version-pointer changes**

Validate candidate ID, manifest, benchmark eligibility, and approval before writing. Store `active_model`, `previous_model`, candidate metadata, and promotion timestamp as one JSON document using temp-file replacement. Rollback changes only the pointer; retain all versioned model tags. Resolve `tuminh-active` only when the active registry entry exists and references an installed candidate.

- [ ] **Step 4: Run registry tests**

Run: `python -m pytest tests/test_distill_registry.py -q`
Expected: PASS; rejected promotion and failed writes do not change the active model.

- [ ] **Step 5: Commit**

```bash
git add distill/registry.py tests/test_distill_registry.py
git commit -m "feat: gate model promotion and rollback"
```

## Task 6: Add optional QLoRA trainer and measured GPU profiles

**Files:**
- Create: `distill/trainer.py`
- Create: `distill/config.example.toml`
- Create: `distill/requirements-training.txt`
- Create: `tests/test_distill_trainer.py`
- Modify: `tests/distill_fakes.py`

**Interfaces:**
- `TrainingConfig(model_id: str, model_revision: str, license_or_authorization: str, quantization: str, sequence_length: int, batch_size: int, gradient_accumulation_steps: int, lora_rank: int, output_path: Path, hardware_profile: str | None)`
- `TrainingConfig.from_toml(path: Path) -> TrainingConfig`
- `Trainer.train(dataset_path: Path, config: TrainingConfig) -> CandidateManifest`
- `HardwareProbe.measure_peak_vram() -> int | None`
- `tests/distill_fakes.py` defines `valid_training_config(output_root: Path) -> TrainingConfig` and `FakeTrainingBackend(cuda_available: bool)`.

- [ ] **Step 1: Write CPU-only config and fake-trainer tests**

Test invalid sequence length, batch size, LoRA rank, missing model-license metadata, missing dataset, unavailable CUDA, and successful invocation of an injected fake training backend. Assert unavailable/incompatible hardware fails explicitly and never changes the model registry.

```python
def test_trainer_reports_missing_cuda_without_registry_changes(tmp_path):
    trainer = Trainer(backend=FakeTrainingBackend(cuda_available=False))
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{\"messages\": []}\n", encoding="utf-8")
    with pytest.raises(TrainingEnvironmentError):
        trainer.train(dataset, valid_training_config(tmp_path))
    assert not (tmp_path / "registry.json").exists()
```

- [ ] **Step 2: Run trainer tests and verify the expected failure**

Run: `python -m pytest tests/test_distill_trainer.py -q`
Expected: FAIL because `distill.trainer` does not exist.

- [ ] **Step 3: Implement an optional, injectable QLoRA backend**

Keep training dependencies out of `pyproject.toml` runtime/dev dependencies; document a separate compatible Linux/WSL2 installation in `distill/requirements-training.txt`. Validate config and dataset before loading the backend. Use configurable model/revision and LoRA settings; collect peak VRAM when available. The optional RTX 3060 Ti profile may lower sequence/batch settings or raise a typed out-of-memory error; never report fixed memory or speed guarantees. Store adapters/checkpoints only in ignored candidate paths.

- [ ] **Step 4: Run CPU-only trainer tests**

Run: `python -m pytest tests/test_distill_trainer.py -q`
Expected: PASS with fake backends; no CUDA package, model download, or network is required.

- [ ] **Step 5: Commit**

```bash
git add distill/trainer.py distill/config.example.toml distill/requirements-training.txt tests/test_distill_trainer.py
git commit -m "feat: add optional local qlora training adapter"
```

## Task 7: Add versioned GGUF/Ollama candidate deployment

**Files:**
- Create: `distill/deploy.py`
- Create: `tests/test_distill_deploy.py`
- Modify: `tests/distill_fakes.py`

**Interfaces:**
- `ModelConverter.convert(adapter_path: Path, output_path: Path, config: ConversionConfig) -> Path`
- `OllamaAdapter.install_candidate(model_path: Path, candidate_id: str) -> str`
- `OllamaAdapter.is_installed(model_tag: str) -> bool`
- `OllamaError` is a typed deployment-adapter error.
- `CandidateDeployer.install(manifest: CandidateManifest, adapter_path: Path) -> str`
- `CandidateDeployer(converter: ModelConverter, ollama: OllamaAdapter, registry: DistillModelRegistry)`
- `tests/distill_fakes.py` defines `FakeConverter()` and `FakeOllamaAdapter(install_error: OllamaError | None = None)`.
- Deployment installs a versioned candidate only; it never changes the active registry pointer.

- [ ] **Step 1: Write fake converter/Ollama tests**

Test a successful candidate receives a versioned tag, failed conversion leaves no registered candidate, failed Ollama installation preserves all active/previous registry state, and model tags are derived from validated candidate IDs.

```python
def test_failed_candidate_install_does_not_change_active_pointer(tmp_path):
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active("baseline-v1")
    deployer = CandidateDeployer(
        converter=FakeConverter(),
        ollama=FakeOllamaAdapter(install_error=OllamaError("offline")),
        registry=registry,
    )
    with pytest.raises(DeploymentError):
        deployer.install(valid_candidate_manifest("candidate-v2"), tmp_path / "adapter")
    assert registry.read().active_model == "baseline-v1"
```

- [ ] **Step 2: Run deployment tests and verify the expected failure**

Run: `python -m pytest tests/test_distill_deploy.py -q`
Expected: FAIL because `distill.deploy` does not exist.

- [ ] **Step 3: Implement candidate-only conversion and installation**

Invoke conversion/Ollama through narrow injected adapters and fixed argv operations. Use versioned tags; never overwrite or promote the active model as part of training/deployment. Persist candidate metadata only after conversion and installation succeed. Keep previous tags available for rollback.

- [ ] **Step 4: Run deployment tests**

Run: `python -m pytest tests/test_distill_deploy.py -q`
Expected: PASS with fakes; every failure leaves the active model pointer unchanged.

- [ ] **Step 5: Commit**

```bash
git add distill/deploy.py tests/test_distill_deploy.py
git commit -m "feat: deploy versioned local model candidates"
```

## Task 8: Add distillation CLI, runtime model-name resolver, and documentation

**Files:**
- Modify: `distill/cli.py`
- Create: `tests/test_distill_cli.py`
- Modify: `distill/registry.py`
- Modify: `agent/model_client.py`
- Modify: `agent/config.py`
- Modify: `.env.example`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Commands: `export-runtime`, `import-external`, `curate`, `train`, `benchmark`, `promote`, `rollback`, and `purge`
- `DistillModelRegistry.resolve_active_model_name("tuminh-active") -> str`
- `OpenAICompatibleModelClient` accepts an injected `ModelNameResolver`; ordinary model names remain unchanged.

- [ ] **Step 1: Write CLI, resolver, and no-network tests**

Test parser help for every command, external import requiring source/license/operator fields, failed eval refusing promotion, negative human confirmation preserving active tag, successful confirmation moving the pointer, rollback restoring previous tag, and `MODEL_NAME=tuminh-active` resolving to the registry's versioned model.

```python
def test_active_model_sentinel_uses_registry_version(tmp_path):
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active("tuminh:v2")

    assert registry.resolve_active_model_name("tuminh-active") == "tuminh:v2"
```

- [ ] **Step 2: Run CLI and resolver tests and verify the expected failure**

Run: `python -m pytest tests/test_distill_cli.py -q`
Expected: FAIL because the distillation CLI and active-model resolver are missing.

- [ ] **Step 3: Implement explicit CLI gates and runtime resolver injection**

Wire each CLI action to the corresponding tested service. `promote` must print baseline/candidate benchmark results, obtain affirmative interactive input, create `HumanApproval`, and then atomically update the registry. `rollback` requires an explicit CLI command. Never auto-edit `.env`; document configuring `MODEL_BASE_URL=http://localhost:11434/v1` and `MODEL_NAME=tuminh-active` once. Resolve the sentinel at runtime-client construction and fail clearly if no active registry entry exists. Add `.gitignore` entries for raw/curated data, runtime registry, checkpoints, adapters, GGUFs, and egg-info build output.

- [ ] **Step 4: Run distillation CLI and full CPU suite**

Run: `python -m pytest tests/test_distill_cli.py -q`
Run: `python -m pytest -q`
Run: `python -m distill.cli --help`
Expected: PASS without CUDA, Ollama, provider credentials, or network access.

- [ ] **Step 5: Commit**

```bash
git add distill/cli.py distill/registry.py agent/model_client.py agent/config.py .env.example .gitignore README.md tests/test_distill_cli.py
git commit -m "feat: add model evolution CLI and approval gates"
```

## Final acceptance check

- [ ] Only authorized, observable runtime/manual exports are accepted; private reasoning fields are rejected.
- [ ] Failed or unverified trajectories cannot enter the curated training dataset.
- [ ] Dataset/manifests are redacted, hashed, atomic, and ignored by Git.
- [ ] Baseline and candidate use the same eval suite; a lower pass rate or any failed safety-critical case blocks eligibility.
- [ ] Promotion additionally requires explicit human approval; rollback restores the previous version.
- [ ] Training/deployment/registry failures leave the active model available.
- [ ] CPU import/curation/benchmark/registry tests run without model downloads or network.
- [ ] CUDA training is optional and the RTX 3060 Ti profile reports measured resources without fixed guarantees.
- [ ] `MODEL_NAME=tuminh-active` resolves to the versioned registry entry at runtime startup.
