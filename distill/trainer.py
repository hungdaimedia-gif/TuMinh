from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, field_validator

from distill.schema import CandidateManifest, TrainingRow


class TrainingError(RuntimeError):
    """Base error for optional local model training."""


class TrainingConfigError(TrainingError):
    """Training settings are missing, invalid, or unsafe."""


class TrainingDatasetError(TrainingError):
    """The curated dataset is missing or does not meet training requirements."""


class TrainingEnvironmentError(TrainingError):
    """The requested optional training backend is unavailable."""


class TrainingOutOfMemoryError(TrainingError):
    """The selected training run exceeded the available accelerator memory."""


class TrainingBackend(Protocol):
    @property
    def cuda_available(self) -> bool: ...

    def train(self, dataset_path: Path, config: TrainingConfig) -> dict[str, Any]: ...


class HardwareProbe(Protocol):
    def measure_peak_vram(self) -> int | None: ...


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    license_or_authorization: str = Field(min_length=1)
    quantization: Literal["4bit", "8bit"]
    sequence_length: StrictInt = Field(ge=1, le=8192)
    batch_size: StrictInt = Field(ge=1, le=128)
    gradient_accumulation_steps: StrictInt = Field(ge=1, le=1024)
    lora_rank: StrictInt = Field(ge=1, le=256)
    output_path: Path
    hardware_profile: str | None = None

    @field_validator("hardware_profile")
    @classmethod
    def validate_hardware_profile(cls, value: str | None) -> str | None:
        supported = {"rtx_3060_ti_8gb"}
        if value is not None and value not in supported:
            raise ValueError(
                f"unsupported hardware_profile {value!r}; supported: {', '.join(sorted(supported))}"
            )
        return value

    @classmethod
    def from_toml(cls, path: Path) -> TrainingConfig:
        config_path = Path(path)
        try:
            with config_path.open("rb") as stream:
                document = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise TrainingConfigError(f"Could not read training TOML: {config_path}") from error
        if set(document) != {"training"} or not isinstance(document["training"], dict):
            raise TrainingConfigError("Training TOML must contain only a [training] table")
        settings = document["training"]
        try:
            if "output_path" in settings:
                output_path = Path(settings["output_path"])
                if not output_path.is_absolute():
                    settings = {**settings, "output_path": config_path.parent / output_path}
            return cls.model_validate(settings)
        except (ValidationError, TypeError, ValueError) as error:
            raise TrainingConfigError("Training TOML settings are invalid") from error


class OptionalTorchHardwareProbe:
    """Measures accelerator peak allocation only when optional PyTorch is installed."""

    def reset_peak_measurement(self) -> None:
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def measure_peak_vram(self) -> int | None:
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())


class QLoRABackend:
    """Explicitly invoked Hugging Face QLoRA implementation with lazy optional imports."""

    @property
    def cuda_available(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return bool(torch.cuda.is_available())

    def train(self, dataset_path: Path, config: TrainingConfig) -> dict[str, Any]:
        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                DataCollatorForLanguageModeling,
                Trainer as TransformersTrainer,
                TrainingArguments,
            )
        except ImportError as error:
            raise TrainingEnvironmentError(
                "QLoRA dependencies are unavailable; install distill/requirements-training.txt "
                "in the optional Linux/WSL2 training environment"
            ) from error

        rows = [
            TrainingRow.model_validate_json(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        quantization = BitsAndBytesConfig(
            load_in_4bit=config.quantization == "4bit",
            load_in_8bit=config.quantization == "8bit",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.model_revision,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            quantization_config=quantization,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_rank * 2,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        examples = [
            {
                "text": "\n".join(
                    f"{message.role}: {message.content}"
                    for message in row.messages
                )
            }
            for row in rows
        ]
        dataset = Dataset.from_list(examples)

        def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            return tokenizer(
                batch["text"],
                max_length=config.sequence_length,
                truncation=True,
                padding="max_length",
            )

        dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
        dataset = dataset.map(lambda batch: {"labels": batch["input_ids"]}, batched=True)
        arguments = TrainingArguments(
            output_dir=str(config.output_path),
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=True,
            logging_strategy="epoch",
            save_strategy="epoch",
            report_to=[],
            remove_unused_columns=False,
        )
        trainer = TransformersTrainer(
            model=model,
            args=arguments,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()
        model.save_pretrained(config.output_path)
        tokenizer.save_pretrained(config.output_path)
        return {
            "backend": "huggingface-peft-qlora",
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
        }


class Trainer:
    _CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
    _RTX_SEQUENCE_LIMIT = 2048
    _RTX_BATCH_LIMIT = 1

    def __init__(
        self,
        backend: TrainingBackend | None = None,
        *,
        candidate_root: Path | None = None,
        hardware_probe: HardwareProbe | None = None,
    ):
        self.backend = backend if backend is not None else QLoRABackend()
        self.candidate_root = (
            Path(candidate_root)
            if candidate_root is not None
            else Path(__file__).parent / "models" / "candidates"
        )
        self.hardware_probe = hardware_probe or OptionalTorchHardwareProbe()

    def train(self, dataset_path: Path, config: TrainingConfig) -> CandidateManifest:
        validated_config = self._validate_config(config)
        rows, dataset_bytes = self._load_dataset(Path(dataset_path))
        del rows
        self._validate_hardware_profile(validated_config)
        self._require_cuda_if_needed(validated_config)
        output_path = self._validate_output_path(validated_config.output_path)
        if output_path.exists():
            raise TrainingConfigError(f"Candidate output already exists: {output_path}")

        candidate_root = self.candidate_root.resolve(strict=False)
        candidate_root.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=candidate_root)
        )
        staged_config = validated_config.model_copy(update={"output_path": stage})
        try:
            dataset_snapshot = stage / ".training-dataset.jsonl"
            dataset_snapshot.write_bytes(dataset_bytes)
            reset_probe = getattr(self.hardware_probe, "reset_peak_measurement", None)
            if callable(reset_probe):
                reset_probe()
            try:
                backend_metadata = self.backend.train(
                    dataset_snapshot,
                    staged_config,
                )
            except TrainingError:
                raise
            except MemoryError as error:
                raise TrainingOutOfMemoryError(
                    "Training ran out of accelerator memory; lower sequence_length or batch_size"
                ) from error
            except Exception as error:
                if "out of memory" in str(error).lower():
                    raise TrainingOutOfMemoryError(
                        "Training ran out of accelerator memory; lower sequence_length or batch_size"
                    ) from error
                raise TrainingError("Training backend failed before producing a candidate") from error
            if not isinstance(backend_metadata, dict):
                raise TrainingError("Training backend returned invalid runtime metadata")
            dataset_snapshot.unlink(missing_ok=True)
            self._validate_staged_artifacts(stage)
            peak_vram = self.hardware_probe.measure_peak_vram()
            if peak_vram is not None and (
                not isinstance(peak_vram, int)
                or isinstance(peak_vram, bool)
                or peak_vram < 0
            ):
                raise TrainingError("Hardware probe returned an invalid peak VRAM measurement")
            now = datetime.now(UTC).isoformat()
            runtime_metadata = {
                **backend_metadata,
                "training_completed": True,
                "training_finished_at": now,
                "benchmark_status": "pending",
                "python_version": sys.version.split()[0],
            }
            hardware_metadata = {
                "hardware_profile": validated_config.hardware_profile,
                "cuda_available": True,
                "peak_vram_bytes": peak_vram,
                "measurement_source": (
                    type(self.hardware_probe).__name__ if peak_vram is not None else None
                ),
            }
            manifest = CandidateManifest(
                candidate_id=output_path.name,
                model_id=validated_config.model_id,
                model_revision=validated_config.model_revision,
                base_model_license=validated_config.license_or_authorization,
                dataset_hash=hashlib.sha256(dataset_bytes).hexdigest(),
                training_config=validated_config.model_dump(mode="json"),
                runtime_metadata=runtime_metadata,
                hardware_metadata=hardware_metadata,
                benchmark_id=f"training-reserved-{uuid.uuid4().hex}",
            )
            manifest_path = stage / "candidate-manifest.json"
            manifest_path.write_text(
                manifest.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(stage, output_path)
            return manifest
        except TrainingError:
            raise
        except Exception as error:
            raise TrainingError("Could not finalize the candidate training artifacts") from error
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)

    @staticmethod
    def _validate_config(config: TrainingConfig) -> TrainingConfig:
        try:
            return TrainingConfig.model_validate(
                config.model_dump(mode="python")
                if isinstance(config, TrainingConfig)
                else config
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise TrainingConfigError("Training configuration is invalid") from error

    @staticmethod
    def _load_dataset(dataset_path: Path) -> tuple[list[TrainingRow], bytes]:
        if dataset_path.is_symlink():
            raise TrainingDatasetError("Curated dataset must not be a symlink")
        if not dataset_path.exists() or not dataset_path.is_file():
            raise TrainingDatasetError(f"Curated dataset does not exist: {dataset_path}")
        try:
            dataset_bytes = dataset_path.read_bytes()
            text = dataset_bytes.decode("utf-8")
        except (OSError, UnicodeError) as error:
            raise TrainingDatasetError("Could not read curated dataset as UTF-8") from error
        if not text.strip():
            raise TrainingDatasetError("Curated dataset is empty")
        rows: list[TrainingRow] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise TrainingDatasetError(
                    f"Curated dataset contains an empty row at line {line_number}"
                )
            try:
                payload = json.loads(line)
                if (
                    not isinstance(payload, dict)
                    or type(payload.get("verification_passed")) is not bool
                    or type(payload.get("eval_passed")) is not bool
                ):
                    raise ValueError("verification and eval fields must be booleans")
                row = TrainingRow.model_validate_json(line)
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                raise TrainingDatasetError(
                    f"Curated dataset row {line_number} is invalid"
                ) from error
            if row.verification_passed is not True:
                raise TrainingDatasetError(
                    f"Curated dataset row {line_number} has not passed verification"
                )
            if row.eval_passed is not True:
                raise TrainingDatasetError(
                    f"Curated dataset row {line_number} has not passed evaluation"
                )
            rows.append(row)
        if not rows:
            raise TrainingDatasetError("Curated dataset contains no training rows")
        return rows, dataset_bytes

    def _validate_hardware_profile(self, config: TrainingConfig) -> None:
        if config.hardware_profile == "rtx_3060_ti_8gb" and (
            config.sequence_length > self._RTX_SEQUENCE_LIMIT
            or config.batch_size > self._RTX_BATCH_LIMIT
        ):
            raise TrainingOutOfMemoryError(
                "The rtx_3060_ti_8gb profile is conservative, not a VRAM guarantee; "
                "use sequence_length <= 2048 and batch_size = 1, or adjust the profile"
            )

    def _require_cuda_if_needed(self, config: TrainingConfig) -> None:
        try:
            available = self.backend.cuda_available
            if callable(available):
                available = available()
        except TrainingError:
            raise
        except Exception as error:
            raise TrainingEnvironmentError("Could not determine CUDA availability") from error
        if config.quantization in {"4bit", "8bit"} and not available:
            raise TrainingEnvironmentError(
                "CUDA is required for the configured QLoRA quantization; "
                "install the optional training stack and use a compatible GPU"
            )

    @staticmethod
    def _find_git_root(path: Path) -> Path | None:
        probe = path.resolve(strict=False)
        marker_root = next(
            (
                candidate
                for candidate in (probe, *probe.parents)
                if os.path.lexists(candidate / ".git")
            ),
            None,
        )
        if marker_root is None:
            return None
        try:
            result = subprocess.run(
                ["git", "-C", str(marker_root), "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            raise TrainingConfigError(
                "Could not identify Git repository for candidate output; refusing to write under repo paths"
            ) from error
        root = result.stdout.strip()
        if result.returncode != 0 or not root:
            raise TrainingConfigError(
                "Could not identify Git repository for candidate output; refusing to write under repo paths"
            )
        repo_root = Path(root).resolve(strict=False)
        if repo_root != marker_root.resolve(strict=False):
            raise TrainingConfigError(
                "Could not identify Git repository for candidate output; refusing to write under repo paths"
            )
        return repo_root

    @staticmethod
    def _require_git_ignored(target: Path, repo_root: Path) -> None:
        try:
            relative_target = target.resolve(strict=False).relative_to(repo_root.resolve(strict=False)).as_posix()
        except ValueError as error:
            raise TrainingConfigError(
                "Candidate output must remain inside the repository before ignore checks"
            ) from error
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "check-ignore",
                    "--quiet",
                    str(relative_target),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            raise TrainingConfigError(
                "Could not verify candidate output path is gitignored; refusing to write under repo paths"
            ) from error
        if result.returncode == 0:
            return
        if result.returncode == 1:
            raise TrainingConfigError(
                "Candidate output path must be gitignored inside the repository before training can write artifacts"
            )
        raise TrainingConfigError(
            "Could not verify candidate output path is gitignored; refusing to write under repo paths"
        )

    def _validate_output_path(self, output_path: Path) -> Path:
        configured_path = Path(output_path)
        if configured_path.name in {"", ".", ".."} or not self._CANDIDATE_ID.fullmatch(
            configured_path.name
        ):
            raise TrainingConfigError("Candidate output must use a safe candidate ID")
        if self.candidate_root.name not in {"candidates", "checkpoints"}:
            raise TrainingConfigError(
                "Candidate root must be a validated ignored candidates/ or checkpoints/ directory"
            )
        root = self.candidate_root.resolve(strict=False)
        target = configured_path.resolve(strict=False)
        try:
            relative = target.relative_to(root)
        except ValueError as error:
            raise TrainingConfigError(
                "Candidate output must remain beneath the validated candidates/ directory"
            ) from error
        if len(relative.parts) != 1:
            raise TrainingConfigError(
                "Candidate output must be a direct child of the candidates/ directory"
            )
        for path in (configured_path, *configured_path.parents, self.candidate_root, *self.candidate_root.parents):
            if path.exists() and path.is_symlink():
                raise TrainingConfigError("Candidate output path must not traverse a symlink")
        if target.exists():
            raise TrainingConfigError(f"Candidate output already exists: {target}")
        repo_root = self._find_git_root(root)
        if repo_root is not None:
            self._require_git_ignored(target, repo_root)
        return target

    @staticmethod
    def _validate_staged_artifacts(stage: Path) -> None:
        if not stage.is_dir() or not any(stage.iterdir()):
            raise TrainingError("Training backend produced no candidate artifacts")
        for path in stage.rglob("*"):
            if path.is_symlink():
                raise TrainingError("Training backend produced a symlinked candidate artifact")
