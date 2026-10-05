from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import ValidationError

from agent.config import Settings
from distill.benchmark import BenchmarkRunner, ModelEvaluator
from distill.curator import Curator
from distill.deploy import (
    CandidateDeployer,
    ConversionConfig,
    SubprocessModelConverter,
    SubprocessOllamaAdapter,
)
from distill.importers import (
    DistillImportError,
    ExternalImporter,
    RuntimeExporter,
    write_trajectory_exports,
)
from distill.registry import (
    DistillModelRegistry,
    HumanApproval,
)
from distill.schema import (
    BenchmarkMetadata,
    BenchmarkReport,
    CandidateManifest,
    EvalCase,
    TrajectoryExport,
    parse_trajectory_jsonl,
)
from distill.trainer import (
    OptionalTorchHardwareProbe,
    QLoRABackend,
    Trainer,
    TrainingConfig,
)


class DistillationCommandError(ValueError):
    """Raised when a CLI input or service result is invalid."""


@dataclass(slots=True)
class DistillServices:
    registry: DistillModelRegistry | None = None
    curator: Curator | None = None
    trainer: Trainer | None = None
    deployer: CandidateDeployer | None = None
    evaluator_factory: Callable[[str], ModelEvaluator] | None = None


def main(
    argv: Sequence[str] | None = None,
    *,
    services: DistillServices | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.error("A distillation command is required")
    active_services = services or DistillServices()
    try:
        command = arguments.command
        if command == "export-runtime":
            output = RuntimeExporter(
                arguments.journal_root,
                secret_values=_configured_secrets(),
            ).export(arguments.task_id, arguments.output)
            print(f"Exported runtime trajectory to {output}")
        elif command == "import-external":
            if _resolve(arguments.path) == _resolve(arguments.output):
                raise DistillImportError(
                    f"Import output cannot overwrite the supplied source file: {arguments.path}"
                )
            trajectories = ExternalImporter(
                secret_values=_configured_secrets(),
            ).import_file(
                arguments.path,
                source_name=arguments.source_name,
                license_or_authorization=arguments.license_or_authorization,
                authorized_by=arguments.authorized_by,
                source_model_id=arguments.source_model_id,
                source_model_revision=arguments.source_model_revision,
            )
            output = write_trajectory_exports(trajectories, arguments.output)
            print(f"Imported {len(trajectories)} authorized trajectory record(s) to {output}")
        elif command == "curate":
            records = []
            for input_path in arguments.inputs:
                try:
                    records.extend(
                        _parse_export_records(
                            _read_text(input_path, "trajectory export")
                        )
                    )
                except (ValidationError, ValueError) as error:
                    raise DistillImportError(
                        f"Trajectory export is invalid: {input_path}"
                    ) from error
            manifest = _curator(active_services).curate(records, arguments.output)
            print(
                f"Curated {manifest.row_count} row(s) from "
                f"{len(manifest.source_hashes)} source(s) to {arguments.output}"
            )
        elif command == "train":
            config = TrainingConfig.from_toml(arguments.config)
            candidate = _trainer(active_services, arguments.candidate_root).train(
                arguments.dataset,
                config,
            )
            manifest_path = config.output_path / "candidate-manifest.json"
            _write_text(manifest_path, candidate.model_dump_json(indent=2) + "\n")
            print(f"Training candidate awaits benchmark and deployment: {manifest_path}")
        elif command == "deploy":
            manifest = _read_model(arguments.manifest, CandidateManifest, "candidate manifest")
            deployer = _deployer(active_services, arguments)
            installed_tag = deployer.install(manifest, arguments.adapter)
            print(f"Deployed candidate {installed_tag}; active model was not changed")
        elif command == "benchmark":
            report = _benchmark(active_services, arguments)
            _write_text(arguments.output, report.model_dump_json(indent=2) + "\n")
            status = "eligible" if report.eligible_for_promotion else "not eligible"
            print(
                f"Benchmark {report.metadata.benchmark_id} completed "
                f"({status}); report: {arguments.output}"
            )
        elif command == "init-active":
            _registry(active_services, arguments.registry).initialize_active(
                arguments.model_tag,
                baseline_model_id=arguments.baseline_model_id,
                baseline_model_revision=arguments.baseline_model_revision,
            )
            print(
                f"Initialized active model {arguments.model_tag} with baseline identity "
                f"{arguments.baseline_model_id}@{arguments.baseline_model_revision}"
            )
        elif command == "promote":
            registry = _registry(active_services, arguments.registry)
            report = _read_model(arguments.report, BenchmarkReport, "benchmark report")
            _display_report(arguments.candidate_id, report)
            if input_fn(
                f"Promote {arguments.candidate_id} to active? Type yes to continue: "
            ).strip().lower() not in {"yes", "y"}:
                print("Promotion cancelled; active model was not changed")
                return 0
            approved_by = input_fn("Approved by (name or identifier): ").strip()
            if not approved_by:
                raise DistillationCommandError("approved_by must not be empty")
            registry.promote(
                arguments.candidate_id,
                report=report,
                approval=HumanApproval(
                    approved_by=approved_by,
                    confirmed_at=datetime.now(UTC),
                ),
            )
            print(f"Promoted {arguments.candidate_id}; previous active model retained for rollback")
        elif command == "rollback":
            state = _registry(active_services, arguments.registry).rollback()
            print(f"Rolled back active model to {state.active_model}")
        elif command == "purge":
            result = _curator(active_services).purge(
                arguments.source_sha256,
                raw_root=arguments.raw_root,
                curated_root=arguments.curated_root,
            )
            print(
                f"Purged source {result.source_sha256}: {result.raw_files_removed} raw file(s), "
                f"{result.curated_rows_removed} curated row(s) across "
                f"{result.datasets_updated} dataset(s)"
            )
        else:  # pragma: no cover - argparse choices and dispatch remain paired
            raise DistillationCommandError(f"Unsupported command: {command}")
    except Exception as error:
        print(f"Distillation command failed: {error}", file=sys.stderr)
        return 2
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicit, local trajectory distillation and model evolution."
    )
    commands = parser.add_subparsers(dest="command")

    export_runtime = commands.add_parser("export-runtime")
    export_runtime.add_argument(
        "--journal-root",
        type=Path,
        default=Path("memory") / "trajectories",
    )
    export_runtime.add_argument("--task-id", required=True)
    export_runtime.add_argument("--output", type=Path, required=True)

    import_external = commands.add_parser("import-external")
    import_external.add_argument("path", type=Path)
    import_external.add_argument("--source-name", required=True)
    import_external.add_argument("--license-or-authorization", required=True)
    import_external.add_argument("--authorized-by", required=True)
    import_external.add_argument("--source-model-id")
    import_external.add_argument("--source-model-revision")
    import_external.add_argument("--output", type=Path, required=True)

    curate = commands.add_parser("curate")
    curate.add_argument("--input", dest="inputs", action="append", type=Path, required=True)
    curate.add_argument("--output", type=Path, required=True)

    train = commands.add_parser(
        "train",
        help="Run explicitly requested optional local QLoRA training; does not deploy.",
    )
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("distill") / "models" / "candidates",
    )

    deploy = commands.add_parser("deploy")
    deploy.add_argument("--manifest", type=Path, required=True)
    deploy.add_argument("--adapter", type=Path, required=True)
    deploy.add_argument("--registry", type=Path)
    deploy.add_argument("--candidate-root", type=Path, default=Path("distill/models/candidates"))
    deploy.add_argument("--converter-script", type=Path)
    deploy.add_argument("--base-model-path", type=Path)
    deploy.add_argument(
        "--quantization",
        choices=("f16", "Q4_K_M", "Q5_K_M", "Q8_0"),
        default="Q4_K_M",
    )
    deploy.add_argument("--ollama", default="ollama")

    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--candidate-manifest", type=Path, required=True)
    benchmark.add_argument("--registry", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument(
        "--evaluator",
        required=True,
        help="Explicit local evaluator plugin in module:factory form.",
    )
    benchmark.add_argument("--eval-set", type=Path)
    benchmark.add_argument("--tool-limits-json", default='{"max_tool_calls":20}')
    benchmark.add_argument("--decoding-config-json", default='{"temperature":0}')
    benchmark.add_argument("--software-metadata-json", default='{"client":"tuminh-distill-cli"}')
    benchmark.add_argument("--hardware-metadata-json", default='{"execution":"local"}')
    benchmark.add_argument("--eval-set-version", default="tuminh-core-v1")

    initialize = commands.add_parser("init-active")
    initialize.add_argument("--registry", type=Path)
    initialize.add_argument("--model-tag", required=True)
    initialize.add_argument("--baseline-model-id", required=True)
    initialize.add_argument("--baseline-model-revision", required=True)

    promote = commands.add_parser("promote")
    promote.add_argument("--registry", type=Path)
    promote.add_argument("--candidate-id", required=True)
    promote.add_argument("--report", type=Path, required=True)

    rollback = commands.add_parser("rollback")
    rollback.add_argument("--registry", type=Path)

    purge = commands.add_parser("purge")
    purge.add_argument("--source-sha256", required=True)
    purge.add_argument("--raw-root", type=Path, required=True)
    purge.add_argument("--curated-root", type=Path, required=True)
    return parser


def _configured_secrets() -> tuple[str, ...]:
    api_key = os.environ.get("MODEL_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = str(dotenv_values(".env").get("MODEL_API_KEY") or "").strip()
        except (OSError, UnicodeError):
            api_key = ""
    return (api_key,) if api_key else ()


def _registry(
    services: DistillServices,
    path: Path | None,
) -> DistillModelRegistry:
    if services.registry is not None:
        return services.registry
    dotenv = {
        key: value
        for key, value in dotenv_values(".env").items()
        if isinstance(value, str)
    }
    dotenv.update(os.environ)
    configured_path = path or Settings.from_env(dotenv).distill_registry_path
    return DistillModelRegistry(configured_path)


def _curator(services: DistillServices) -> Curator:
    return services.curator or Curator(_configured_secrets())


def _trainer(services: DistillServices, candidate_root: Path) -> Trainer:
    return services.trainer or Trainer(
        backend=QLoRABackend(),
        candidate_root=candidate_root,
        hardware_probe=OptionalTorchHardwareProbe(),
    )


def _deployer(services: DistillServices, arguments: argparse.Namespace) -> CandidateDeployer:
    if services.deployer is not None:
        return services.deployer
    if arguments.converter_script is None or arguments.base_model_path is None:
        raise DistillationCommandError(
            "deploy requires --converter-script and --base-model-path for local conversion"
        )
    registry = _registry(services, arguments.registry)
    return CandidateDeployer(
        converter=SubprocessModelConverter(arguments.converter_script),
        ollama=SubprocessOllamaAdapter(arguments.ollama),
        registry=registry,
        conversion_config=ConversionConfig(
            base_model_path=arguments.base_model_path,
            quantization=arguments.quantization,
        ),
        candidate_root=arguments.candidate_root,
    )


def _benchmark(
    services: DistillServices,
    arguments: argparse.Namespace,
) -> BenchmarkReport:
    registry = _registry(services, arguments.registry)
    state = registry.read()
    if state.active_model is None:
        raise DistillationCommandError("Active model is not initialized in the registry")
    candidate = _read_model(
        arguments.candidate_manifest,
        CandidateManifest,
        "candidate manifest",
    )
    registered = state.candidates.get(candidate.candidate_id)
    if registered is None or registered.model_dump(mode="json") != candidate.model_dump(mode="json"):
        raise DistillationCommandError(
            "Candidate must be deployed and registered before benchmarking"
        )
    active_manifest = state.candidates.get(state.active_model)
    if active_manifest is None:
        baseline_model_id = state.initialized_baseline_model_id
        baseline_revision = state.initialized_baseline_revision
    else:
        baseline_model_id = active_manifest.model_id
        baseline_revision = active_manifest.model_revision
    if not baseline_model_id or not baseline_revision:
        raise DistillationCommandError(
            "Registry does not contain the exact baseline model ID and revision"
        )

    evaluator = (
        services.evaluator_factory(arguments.evaluator)
        if services.evaluator_factory is not None
        else _load_evaluator(arguments.evaluator)
    )
    eval_set_path = arguments.eval_set or Path(__file__).parent / "evals" / "cases.json"
    try:
        raw_cases = json.loads(_read_text(eval_set_path, "evaluation case set"))
        if not isinstance(raw_cases, list):
            raise ValueError("evaluation case set must be a JSON array")
        cases = [EvalCase.model_validate(item) for item in raw_cases]
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        raise DistillationCommandError("Evaluation case set is invalid") from error
    if not cases:
        raise DistillationCommandError("Evaluation case set must not be empty")

    metadata = BenchmarkMetadata(
        benchmark_id=candidate.benchmark_id,
        baseline_model_id=baseline_model_id,
        baseline_model_revision=baseline_revision,
        candidate_model_id=candidate.model_id,
        candidate_model_revision=candidate.model_revision,
        dataset_hash=candidate.dataset_hash,
        eval_set_version=arguments.eval_set_version,
        safety_critical_case_ids=[
            case.case_id for case in cases if case.safety_critical
        ],
        tool_limits=_json_object(arguments.tool_limits_json, "tool limits"),
        decoding_config=_json_object(arguments.decoding_config_json, "decoding config"),
        software_metadata=_json_object(
            arguments.software_metadata_json, "software metadata"
        ),
        hardware_metadata=_json_object(
            arguments.hardware_metadata_json, "hardware metadata"
        ),
    )
    return BenchmarkRunner(evaluator, cases).compare(
        state.active_model,
        candidate.candidate_id,
        metadata,
    )


def _load_evaluator(specification: str) -> ModelEvaluator:
    if ":" not in specification:
        raise DistillationCommandError("Evaluator must use module:factory syntax")
    module_name, factory_name = specification.split(":", 1)
    if not module_name or not factory_name:
        raise DistillationCommandError("Evaluator must use module:factory syntax")
    try:
        module = importlib.import_module(module_name)
        factory: Any = getattr(module, factory_name)
        evaluator = factory()
    except Exception as error:
        raise DistillationCommandError(
            f"Could not load local evaluator plugin {specification!r}"
        ) from error
    if not callable(getattr(evaluator, "evaluate", None)):
        raise DistillationCommandError(
            "Evaluator factory must return an object implementing evaluate(model_name, cases, *, "
            "tool_limits, decoding_config)"
        )
    return evaluator


def _parse_export_records(content: str) -> list[TrajectoryExport]:
    records: list[TrajectoryExport] = []
    current_lines: list[str] = []
    for line in content.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise DistillImportError("Trajectory export contains invalid JSON") from error
        if isinstance(value, dict) and value.get("record_type") == "header":
            if current_lines:
                records.append(parse_trajectory_jsonl("\n".join(current_lines)))
            current_lines = [line]
        elif not current_lines:
            raise DistillImportError("Trajectory export must start with a header record")
        else:
            current_lines.append(line)
    if current_lines:
        records.append(parse_trajectory_jsonl("\n".join(current_lines)))
    if not records:
        raise DistillImportError("Trajectory export must not be empty")
    return records


def _display_report(candidate_id: str, report: BenchmarkReport) -> None:
    print(
        f"Promotion review for {candidate_id}: "
        f"baseline={report.baseline_score.pass_rate:.3f} "
        f"candidate={report.candidate_score.pass_rate:.3f} "
        f"eligible={report.eligible_for_promotion}"
    )
    print(f"Safety outcomes: {json.dumps(report.safety_results, sort_keys=True)}")


def _json_object(raw: str, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DistillationCommandError(f"{name} must be a JSON object") from error
    if not isinstance(value, dict) or not value:
        raise DistillationCommandError(f"{name} must be a non-empty JSON object")
    return value


def _read_model(path: Path, model_type: type[Any], description: str) -> Any:
    try:
        return model_type.model_validate_json(_read_text(path, description))
    except (ValidationError, ValueError) as error:
        raise DistillationCommandError(f"{description.capitalize()} is invalid: {path}") from error


def _read_text(path: Path, description: str) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise DistillationCommandError(
            f"{description.capitalize()} is not a regular local file: {source}"
        )
    try:
        return source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise DistillationCommandError(f"Could not read {description}: {source}") from error


def _write_text(path: Path, content: str) -> Path:
    destination = Path(path)
    if destination.is_symlink():
        raise DistillationCommandError(f"Refusing to overwrite a symlink: {destination}")
    temporary_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except OSError as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise DistillationCommandError(f"Could not write local output: {destination}") from error
    return destination


def _resolve(path: Path) -> Path:
    return Path(path).resolve(strict=False)


if __name__ == "__main__":
    raise SystemExit(main())
