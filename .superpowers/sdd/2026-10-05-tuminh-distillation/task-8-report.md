# Task 8 implementation report

## Implementation

- Added the explicit local CLI commands `curate`, `train`, `deploy`,
  `benchmark`, `init-active`, `promote`, `rollback`, and `purge`, preserving
  `export-runtime` and `import-external`.
- Connected CLI operations to the existing curator, trainer, deployment,
  benchmark, and registry services. Training does not deploy or promote;
  deployment registers only after service success; promotion displays scores
  and safety outcomes, requires explicit affirmative confirmation and records
  the approver; purge delegates exact-hash removal to the curator.
- Benchmarking uses the reserved candidate benchmark ID, requires the
  registered candidate and exact baseline identity, defaults to the bundled
  eval case set, and imports an evaluator plugin only for an explicitly run
  benchmark command. CLI tests inject local fakes and do not use network,
  Ollama, or optional training packages.
- Added `DISTILL_REGISTRY_PATH` to settings and `.env.example`. Runtime
  construction resolves `tuminh-active` through the local registry before
  constructing the provider client and raises `ModelConfigurationError` for
  an unavailable active model; ordinary model names remain unchanged.
- Documented CPU-only use, optional training setup, authorization/provenance,
  CLI examples, evaluator contract, baseline identity, explicit promotion,
  rollback, and startup-time model resolution in `README.md`.

## TDD and verification

- Added CLI, settings, and model-client tests before implementation. Initial
  focused execution failed on the missing commands/config/resolver as
  expected. Additional benchmark-tag routing and configured-dotenv path tests
  were each run red before their corresponding fixes.
- Focused verification:
  `python -m pytest tests\test_distill_cli.py tests\test_config.py tests\test_model_client.py -q`
  — **47 passed**.
- Full suite: `python -m pytest -q` — **303 passed, 3 skipped**. The skipped
  tests require Windows symlink privileges not available in this environment.
- `python -m distill.cli --help` — succeeded and lists all ten commands.
- `python -m evals.run` — all four deterministic offline evaluations passed.
- `git diff --check` — succeeded.

## Self-review

- Reviewed the changed CLI, resolver/configuration, tests, and README. Changes
  are confined to Task 8 files; previous import, curation, training,
  deployment, benchmark, and registry safety gates were not modified.
- Confirmed plugin loading is deferred until `benchmark` execution and
  successful training does not call deployment or promotion.
- No live model endpoint, GPU training stack, Ollama deployment, or external
  evaluator was invoked. These integrations remain optional and were covered
  with injected local test doubles.

## Review addendum — CPU-only CLI test isolation

- Replaced the real `OptionalTorchHardwareProbe` in the CLI training test with
  an injected deterministic no-op probe whose VRAM measurement is `None`.
- Added a guard that fails if the optional torch probe is constructed, plus
  an assertion that the resulting candidate manifest records no VRAM
  measurement.
- TDD check: restoring the old probe construction made this focused test fail
  at the guard as expected; injecting the no-op fake made it pass.
- Re-ran the focused CLI/config/model-client tests (**47 passed**), the full
  suite (**303 passed, 3 skipped** for unavailable Windows symlink privileges),
  CLI help (all ten commands shown), offline evals (all four passed), and
  `git diff --check` (passed).
- The separately deferred Minor eval-set version-label issue was not changed.
