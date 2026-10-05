# Task 4 implementation report

- Added strict shared evaluation, score, metadata, and benchmark-report schemas; an injected evaluator/runner; and the versioned offline eval-case fixture.
- The runner evaluates baseline and candidate sequentially against the same immutable case snapshot and the same deeply immutable `tool_limits` and `decoding_config`, hashes the pinned cases, and records model revisions, pass counts/rates, safety outcomes, configuration, software/hardware metadata, and resource measurements.
- Eligibility fails closed for incomplete or inconsistent scores/metadata, evaluator errors, missing safety results, failed critical cases, and aggregate regressions. No registry or promotion behavior was added.
- Self-review added regression coverage for missing case outcomes, malformed metadata, untrusted safety-case IDs, coerced result values, and evaluator errors.
- Evaluators now receive both configuration mappings explicitly. The benchmark runner creates private JSON snapshots, recursively freezes mappings and sequences, and passes the exact same objects to both evaluations; attempted mutation is rejected and surfaced as a benchmark error before promotion eligibility can be reported.
- Focused tests: `16 passed`.
- Full CPU suite: `215 passed, 3 skipped`; all skips were existing symlink tests unavailable under this Windows account's symlink permissions.
- `git diff --cached --check`: clean.

## Task 4 review finding follow-up

- Regression proof: both evaluations receive identical settings objects, including nested values; top-level and nested mutation attempts fail, and an evaluator that attempts to change a setting is rejected with `BenchmarkError`.
- Self-review: configuration values are JSON-normalized before freezing so mutable lists become tuples and nested dictionaries become read-only mappings. Invalid/non-JSON settings fail closed. The evaluator protocol now requires both explicit settings arguments.
- Exact validation command: `python -m pytest tests\test_distill_benchmark.py -q`
- Exact output:
  ```text
  ................                                                         [100%]
  16 passed in 0.11s
  ```
- `git diff --check`: clean.

## Task 4 outdated evaluator interface follow-up

- The `ModelEvaluator` Protocol and its method docstring now explicitly require `tool_limits` and `decoding_config` as keyword-only settings; the interim two-argument evaluator signature is not compatible.
- An evaluator that rejects these keyword arguments now fails closed with a `BenchmarkError` that states the required signature and directs implementers to accept both settings. A focused regression test verifies this guidance; shared immutable settings enforcement remains unchanged.
- Regression proof: the new test first failed because the outdated evaluator produced only the generic `Model evaluation failed` error; after the fix it passed with the migration guidance assertion.
- Exact validation command (project `.venv\Scripts` was prepended to `PATH` so `python` resolves the repository environment):
  ```text
  python -m pytest tests\test_distill_benchmark.py -q
  ```
- Exact output:
  ```text
  .................                                                        [100%]
  17 passed in 0.10s
  ```
- Fix self-review: the keyword-only settings are still passed as the same immutable mapping instances to both model evaluations. Legacy signatures are not retried without settings; unsupported keyword arguments are converted to actionable `BenchmarkError`s. No unrelated files or behaviors were changed.
