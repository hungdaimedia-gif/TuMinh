from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from distill.registry import DistillModelRegistry, RegistryError
from distill.schema import CandidateManifest
from distill.trainer import (
    TrainingConfigError,
    validate_candidate_output_path,
)


_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class DeploymentError(RuntimeError):
    """Raised when a candidate cannot be safely converted or installed."""


class ConversionError(DeploymentError):
    """Raised when the local adapter-to-GGUF conversion fails."""


class OllamaError(DeploymentError):
    """Raised when a local Ollama operation fails."""


class ConversionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    base_model_path: Path | None = None
    quantization: Literal["f16", "Q4_K_M", "Q5_K_M", "Q8_0"] = "Q4_K_M"


class ModelConverter(Protocol):
    def convert(
        self,
        adapter_path: Path,
        output_path: Path,
        config: ConversionConfig,
    ) -> Path: ...


class OllamaAdapter(Protocol):
    def install_candidate(self, model_path: Path, candidate_id: str) -> str: ...

    def is_installed(self, model_tag: str) -> bool: ...


class SubprocessModelConverter:
    """Run local llama.cpp conversion tools using argv, never through a shell."""

    def __init__(
        self,
        converter_script: Path,
        *,
        quantizer_executable: str = "llama-quantize",
    ):
        self.converter_script = Path(converter_script)
        self.quantizer_executable = quantizer_executable

    def convert(
        self,
        adapter_path: Path,
        output_path: Path,
        config: ConversionConfig,
    ) -> Path:
        try:
            config = ConversionConfig.model_validate(
                config.model_dump(mode="python")
                if isinstance(config, ConversionConfig)
                else config
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise ConversionError("GGUF conversion settings are invalid") from error
        adapter = _require_directory(adapter_path, "adapter")
        output = Path(output_path)
        if output.is_symlink() or not output.is_dir():
            raise ConversionError("Conversion output must be an existing local directory")
        if config.base_model_path is None:
            raise ConversionError("A local base_model_path is required for GGUF conversion")
        base_model = _require_directory(config.base_model_path, "base model")
        script = Path(self.converter_script)
        if script.is_symlink() or not script.is_file():
            raise ConversionError(f"Local llama.cpp converter script does not exist: {script}")

        unquantized = output / "model-f16.gguf"
        self._run(
            [
                sys.executable,
                str(script.resolve()),
                "--base",
                str(base_model),
                "--outfile",
                str(unquantized),
                "--outtype",
                "f16",
                str(adapter),
            ],
            "llama.cpp adapter conversion",
        )
        _require_file(unquantized, "converter output")
        if config.quantization == "f16":
            return unquantized

        quantized = output / "model.gguf"
        self._run(
            [
                self.quantizer_executable,
                str(unquantized),
                str(quantized),
                config.quantization,
            ],
            "GGUF quantization",
        )
        _require_file(quantized, "quantizer output")
        return quantized

    @staticmethod
    def _run(argv: list[str], operation: str) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )
        except OSError as error:
            raise ConversionError(f"Could not start {operation}") from error
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            suffix = f": {detail[:500]}" if detail else ""
            raise ConversionError(f"{operation} failed{suffix}")
        return result


class SubprocessOllamaAdapter:
    """Create and inspect local Ollama tags using fixed argv operations."""

    def __init__(self, executable: str = "ollama"):
        if not executable or not executable.strip():
            raise OllamaError("Ollama executable must not be empty")
        self.executable = executable

    def install_candidate(self, model_path: Path, candidate_id: str) -> str:
        if not isinstance(candidate_id, str) or not _CANDIDATE_ID.fullmatch(candidate_id):
            raise OllamaError("Candidate tag must be a validated candidate ID")
        model = Path(model_path)
        try:
            model = _require_file(model, "GGUF model")
        except ConversionError as error:
            raise OllamaError("Ollama candidate input must be a local GGUF file") from error
        model_text = str(model)
        if any(char in model_text for char in ('"', "\r", "\n")):
            raise OllamaError("GGUF path cannot be represented safely in an Ollama Modelfile")
        modelfile = model.parent / f".{candidate_id}.Modelfile"
        if modelfile.exists() or modelfile.is_symlink():
            raise OllamaError("Temporary Ollama Modelfile already exists")
        try:
            modelfile.write_text(
                f'FROM "{model}"\n',
                encoding="utf-8",
            )
            self._run(
                [
                    self.executable,
                    "create",
                    candidate_id,
                    "-f",
                    str(modelfile),
                ],
                "create candidate",
            )
        finally:
            modelfile.unlink(missing_ok=True)
        return candidate_id

    def is_installed(self, model_tag: str) -> bool:
        if not isinstance(model_tag, str) or not _CANDIDATE_ID.fullmatch(model_tag):
            raise OllamaError("Model tag must be a validated candidate ID")
        try:
            result = subprocess.run(
                [self.executable, "show", model_tag],
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )
        except OSError as error:
            raise OllamaError("Could not start Ollama to inspect the candidate tag") from error
        return result.returncode == 0

    def _run(self, argv: list[str], operation: str) -> None:
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )
        except OSError as error:
            raise OllamaError(f"Could not start Ollama to {operation}") from error
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            suffix = f": {detail[:500]}" if detail else ""
            raise OllamaError(f"Ollama could not {operation}{suffix}")


class CandidateDeployer:
    def __init__(
        self,
        converter: ModelConverter,
        ollama: OllamaAdapter,
        registry: DistillModelRegistry,
        *,
        conversion_config: ConversionConfig | None = None,
        candidate_root: Path | None = None,
    ):
        self.converter = converter
        self.ollama = ollama
        self.registry = registry
        self.conversion_config = conversion_config or ConversionConfig()
        self.candidate_root = (
            Path(candidate_root)
            if candidate_root is not None
            else Path(__file__).parent / "models" / "candidates"
        )

    def install(self, manifest: CandidateManifest, adapter_path: Path) -> str:
        validated_manifest = self._validate_manifest(manifest)
        adapter = self._validate_adapter(adapter_path, validated_manifest)
        try:
            config = ConversionConfig.model_validate(
                self.conversion_config.model_dump(mode="python")
            )
            registry_state = self.registry.read()
        except (ValidationError, RegistryError, TypeError, ValueError) as error:
            raise DeploymentError("Candidate deployment configuration or registry is invalid") from error
        candidate_id = validated_manifest.candidate_id
        if (
            candidate_id in registry_state.installed_models
            or candidate_id in registry_state.candidates
        ):
            raise DeploymentError(
                "Candidate ID is already installed or registered; refusing to overwrite a version"
            )

        final_output = self.candidate_root / _artifact_directory_name(candidate_id)
        try:
            validated_output = validate_candidate_output_path(
                final_output,
                self.candidate_root,
            )
        except TrainingConfigError as error:
            raise DeploymentError(str(error)) from error

        try:
            if self.ollama.is_installed(candidate_id):
                raise DeploymentError(
                    "Ollama candidate tag is already installed; refusing to overwrite a version"
                )
        except DeploymentError:
            raise
        except Exception as error:
            raise DeploymentError("Could not verify the Ollama candidate tag is unused") from error

        try:
            self.candidate_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise DeploymentError("Could not create candidate output directory") from error
        try:
            validated_output = validate_candidate_output_path(
                final_output,
                self.candidate_root,
            )
        except TrainingConfigError as error:
            raise DeploymentError(str(error)) from error
        try:
            stage = Path(
                tempfile.mkdtemp(
                    prefix=f".{validated_manifest.candidate_id}-gguf.",
                    dir=self.candidate_root,
                )
            )
        except OSError as error:
            raise DeploymentError("Could not create temporary candidate conversion output") from error
        try:
            try:
                converted_path = Path(
                    self.converter.convert(adapter, stage, config)
                )
            except Exception as error:
                if isinstance(error, DeploymentError):
                    raise DeploymentError(f"Candidate conversion failed: {error}") from error
                raise DeploymentError("Candidate conversion failed") from error
            model_path = self._validate_converted_output(converted_path, stage)
            try:
                returned_tag = self.ollama.install_candidate(model_path, candidate_id)
                if returned_tag != candidate_id:
                    raise DeploymentError(
                        "Ollama adapter returned a tag different from the validated candidate ID"
                    )
                if not self.ollama.is_installed(returned_tag):
                    raise DeploymentError(
                        "Ollama did not confirm that the candidate tag is installed"
                    )
            except Exception as error:
                raise DeploymentError(f"Candidate Ollama installation failed: {error}") from error

            if validated_output.exists():
                raise DeploymentError(
                    f"Candidate output already exists: {validated_output}"
                )
            try:
                os.rename(stage, validated_output)
            except OSError as error:
                raise DeploymentError("Could not finalize converted candidate artifacts") from error
            try:
                self.registry.register_candidate(validated_manifest)
            except RegistryError as error:
                raise DeploymentError("Could not register the installed candidate") from error
            return returned_tag
        finally:
            if stage.exists():
                try:
                    shutil.rmtree(stage)
                except OSError as error:
                    raise DeploymentError(
                        "Could not clean incomplete candidate conversion output"
                    ) from error

    @staticmethod
    def _validate_manifest(manifest: CandidateManifest) -> CandidateManifest:
        try:
            return CandidateManifest.model_validate(
                manifest.model_dump(mode="python")
                if isinstance(manifest, CandidateManifest)
                else manifest
            )
        except (ValidationError, AttributeError, TypeError, ValueError) as error:
            raise DeploymentError("Candidate manifest or candidate ID is invalid") from error

    @staticmethod
    def _validate_adapter(
        adapter_path: Path,
        manifest: CandidateManifest,
    ) -> Path:
        source = Path(adapter_path)
        for path in (source, *source.parents):
            if path.exists() and path.is_symlink():
                raise DeploymentError("Adapter input must not traverse a symlink")
        try:
            adapter = source.resolve(strict=True)
        except OSError as error:
            raise DeploymentError(f"Adapter input does not exist: {source}") from error
        if not adapter.is_dir():
            raise DeploymentError("Adapter input must be a directory")
        files = list(adapter.rglob("*"))
        if not files or any(path.is_symlink() for path in files):
            raise DeploymentError("Adapter input must contain files and must not contain symlinks")
        if not any(path.is_file() for path in files):
            raise DeploymentError("Adapter input must contain at least one regular file")

        manifest_path = adapter / "candidate-manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise DeploymentError("Adapter candidate manifest is missing or invalid")
        try:
            stored = CandidateManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as error:
            raise DeploymentError("Adapter candidate manifest is invalid") from error
        if stored.model_dump(mode="json") != manifest.model_dump(mode="json"):
            raise DeploymentError(
                "Adapter candidate manifest does not match the deployment manifest"
            )
        return adapter

    @staticmethod
    def _validate_converted_output(path: Path, stage: Path) -> Path:
        if stage.is_symlink() or any(
            artifact.is_symlink() for artifact in stage.rglob("*")
        ):
            raise DeploymentError("Converted candidate artifacts must not contain symlinks")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(stage.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise DeploymentError(
                "Converter output must be a file inside the temporary candidate directory"
            ) from error
        if not resolved.is_file():
            raise DeploymentError("Converter output must be a regular GGUF file")
        if resolved.suffix.lower() != ".gguf":
            raise DeploymentError("Converter output must be a GGUF file with the .gguf extension")
        return resolved


def _require_directory(path: Path, name: str) -> Path:
    directory = Path(path)
    if directory.is_symlink() or not directory.is_dir():
        raise ConversionError(f"Local {name} directory does not exist: {directory}")
    return directory.resolve(strict=True)


def _require_file(path: Path, name: str) -> Path:
    file_path = Path(path)
    if file_path.is_symlink() or not file_path.is_file():
        raise ConversionError(f"Local {name} file does not exist: {file_path}")
    return file_path.resolve(strict=True)


def _artifact_directory_name(candidate_id: str) -> str:
    suffix = "-gguf"
    if len(candidate_id) + len(suffix) <= 64:
        return candidate_id + suffix
    digest = hashlib.sha256(candidate_id.encode("ascii")).hexdigest()[:8]
    return f"{candidate_id[:50]}-{digest}{suffix}"
