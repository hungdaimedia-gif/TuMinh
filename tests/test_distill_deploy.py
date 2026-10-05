from __future__ import annotations

import json
import multiprocessing
import subprocess
from pathlib import Path

import pytest

import distill.deploy as deploy_module
from distill.schema import CandidateManifest
from distill.deploy import (
    CandidateDeployer,
    ConversionConfig,
    DeploymentError,
    OllamaError,
    SubprocessModelConverter,
    SubprocessOllamaAdapter,
)
from distill.registry import DistillModelRegistry
from tests.distill_fakes import (
    FakeConverter,
    FakeOllamaAdapter,
    valid_candidate_manifest,
)


def make_registry(tmp_path):
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active("baseline-v1")
    return registry


def make_adapter(tmp_path, candidate_id="candidate-v2"):
    adapter_path = tmp_path / "adapter" / candidate_id
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter-config.json").write_text("{}", encoding="utf-8")
    (adapter_path / "candidate-manifest.json").write_text(
        valid_candidate_manifest(candidate_id).model_dump_json(),
        encoding="utf-8",
    )
    return adapter_path


def make_deployer(registry, converter=None, ollama=None, **kwargs):
    kwargs.setdefault("candidate_root", registry.path.parent / "candidates")
    return CandidateDeployer(
        converter=converter or FakeConverter(),
        ollama=ollama or FakeOllamaAdapter(),
        registry=registry,
        **kwargs,
    )


class _ConcurrentConverter:
    def __init__(self, role, first_conversion_started, release_first_conversion):
        self.role = role
        self.first_conversion_started = first_conversion_started
        self.release_first_conversion = release_first_conversion

    def convert(self, adapter_path, output_path, config):
        if self.role == "first":
            self.first_conversion_started.set()
            if not self.release_first_conversion.wait(20):
                raise RuntimeError("test did not release the first conversion")
        manifest = json.loads(
            (adapter_path / "candidate-manifest.json").read_text(encoding="utf-8")
        )
        output_path.mkdir(parents=True, exist_ok=True)
        model_path = output_path / "model.gguf"
        model_path.write_text(manifest["model_revision"], encoding="utf-8")
        return model_path


class _ConcurrentOllamaAdapter:
    def __init__(self, tag_path, role, second_preflight_passed, install_count):
        self.tag_path = Path(tag_path)
        self.role = role
        self.second_preflight_passed = second_preflight_passed
        self.install_count = install_count

    def install_candidate(self, model_path, candidate_id):
        with self.install_count.get_lock():
            self.install_count.value += 1
        self.tag_path.write_bytes(Path(model_path).read_bytes())
        return candidate_id

    def is_installed(self, model_tag):
        installed = self.tag_path.exists()
        if self.role == "second" and not installed:
            self.second_preflight_passed.set()
        return installed


def _run_concurrent_deployment(
    role,
    registry_path,
    candidate_root,
    adapter_path,
    manifest_json,
    tag_path,
    first_conversion_started,
    release_first_conversion,
    second_preflight_passed,
    second_started,
    install_count,
    result_queue,
):
    from distill.deploy import CandidateDeployer
    from distill.registry import DistillModelRegistry

    if role == "second":
        second_started.set()
    deployer = CandidateDeployer(
        converter=_ConcurrentConverter(
            role,
            first_conversion_started,
            release_first_conversion,
        ),
        ollama=_ConcurrentOllamaAdapter(
            tag_path,
            role,
            second_preflight_passed,
            install_count,
        ),
        registry=DistillModelRegistry(Path(registry_path)),
        candidate_root=Path(candidate_root),
    )
    try:
        tag = deployer.install(
            CandidateManifest.model_validate_json(manifest_json),
            Path(adapter_path),
        )
        result_queue.put((role, "installed", tag))
    except Exception as error:
        result_queue.put((role, "rejected", str(error)))


def test_deploys_candidate_tag_after_conversion_and_install_without_promotion(tmp_path):
    registry = make_registry(tmp_path)
    converter = FakeConverter()
    ollama = FakeOllamaAdapter()
    deployer = make_deployer(registry, converter, ollama)
    adapter_path = make_adapter(tmp_path)

    tag = deployer.install(valid_candidate_manifest("candidate-v2"), adapter_path)

    state = registry.read()
    assert tag == "candidate-v2"
    assert len(converter.calls) == 1
    assert converter.calls[0][0] == adapter_path
    assert converter.calls[0][1].parent == tmp_path / "candidates"
    assert converter.calls[0][2] == ConversionConfig()
    assert ollama.installed_tags == ["candidate-v2"]
    assert ollama.is_installed(tag)
    assert set(state.candidates) == {"candidate-v2"}
    assert state.active_model == "baseline-v1"
    assert state.previous_model is None


def test_concurrent_deployments_serialize_per_candidate_across_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    registry = make_registry(tmp_path)
    candidate_id = "candidate-v2"
    manifests = [
        valid_candidate_manifest(candidate_id).model_copy(
            update={"model_revision": f"revision-{role}"}
        )
        for role in ("first", "second")
    ]
    adapters = []
    for index, manifest in enumerate(manifests):
        adapter_path = tmp_path / f"adapter-{index}"
        adapter_path.mkdir()
        (adapter_path / "adapter-config.json").write_text("{}", encoding="utf-8")
        (adapter_path / "candidate-manifest.json").write_text(
            manifest.model_dump_json(),
            encoding="utf-8",
        )
        adapters.append(adapter_path)

    first_conversion_started = context.Event()
    release_first_conversion = context.Event()
    second_preflight_passed = context.Event()
    second_started = context.Event()
    install_count = context.Value("i", 0)
    result_queue = context.Queue()
    tag_path = tmp_path / "ollama" / candidate_id
    tag_path.parent.mkdir()

    def worker(role, adapter_path, manifest):
        return context.Process(
            target=_run_concurrent_deployment,
            args=(
                role,
                str(registry.path),
                str(tmp_path / "candidates"),
                str(adapter_path),
                manifest.model_dump_json(),
                str(tag_path),
                first_conversion_started,
                release_first_conversion,
                second_preflight_passed,
                second_started,
                install_count,
                result_queue,
            ),
        )

    first = worker("first", adapters[0], manifests[0])
    second = worker("second", adapters[1], manifests[1])
    first.start()
    try:
        assert first_conversion_started.wait(15), "first deployment did not reach conversion"
        second.start()
        assert second_started.wait(15), "second deployment did not start"
        second_passed_preflight_while_first_was_converting = second_preflight_passed.wait(2)
    finally:
        release_first_conversion.set()
        first.join(20)
        if second.pid is not None:
            second.join(20)
    for process in (first, second):
        if process.is_alive():
            process.terminate()
            process.join(5)

    assert first.exitcode == 0
    assert second.exitcode == 0
    assert not second_passed_preflight_while_first_was_converting
    outcomes = {
        role: (status, detail)
        for role, status, detail in (result_queue.get(timeout=5) for _ in range(2))
    }
    assert outcomes["first"] == ("installed", candidate_id)
    assert outcomes["second"][0] == "rejected"
    assert "already installed or registered" in outcomes["second"][1]
    assert install_count.value == 1

    state = registry.read()
    assert state.active_model == "baseline-v1"
    assert state.previous_model is None
    assert set(state.candidates) == {candidate_id}
    manifest_revision = state.candidates[candidate_id].model_revision
    artifact = tmp_path / "candidates" / "candidate-v2-gguf" / "model.gguf"
    assert artifact.read_text(encoding="utf-8") == manifest_revision
    assert tag_path.read_text(encoding="utf-8") == manifest_revision
    assert not list((tmp_path / "candidates").glob(".candidate-v2-gguf.*"))


def test_rejects_invalid_candidate_manifest_before_conversion(tmp_path):
    registry = make_registry(tmp_path)
    converter = FakeConverter()
    deployer = make_deployer(registry, converter)
    invalid_manifest = valid_candidate_manifest("candidate-v2").model_copy(
        update={"candidate_id": "bad candidate id"}
    )

    with pytest.raises(DeploymentError, match="manifest|candidate"):
        deployer.install(invalid_manifest, make_adapter(tmp_path))

    assert converter.calls == []
    assert registry.read().candidates == {}


def test_rejects_missing_or_invalid_adapter_before_conversion(tmp_path):
    registry = make_registry(tmp_path)
    converter = FakeConverter()
    deployer = make_deployer(registry, converter)

    with pytest.raises(DeploymentError, match="adapter"):
        deployer.install(
            valid_candidate_manifest("candidate-v2"),
            tmp_path / "missing-adapter",
        )

    assert converter.calls == []
    assert registry.read().candidates == {}


def test_rejects_adapter_manifest_that_does_not_match_before_conversion(tmp_path):
    registry = make_registry(tmp_path)
    converter = FakeConverter()
    deployer = make_deployer(registry, converter)
    adapter_path = make_adapter(tmp_path)
    (adapter_path / "candidate-manifest.json").write_text(
        valid_candidate_manifest("candidate-v3").model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(DeploymentError, match="does not match"):
        deployer.install(valid_candidate_manifest("candidate-v2"), adapter_path)

    assert converter.calls == []
    assert registry.read().candidates == {}


def test_failed_conversion_cleans_only_temporary_output_and_keeps_registry_unchanged(
    tmp_path,
):
    registry = make_registry(tmp_path)
    converter = FakeConverter(failure=RuntimeError("conversion failed"))
    deployer = make_deployer(registry, converter)
    adapter_path = make_adapter(tmp_path)
    previous_artifact = tmp_path / "candidates" / "candidate-v1-gguf" / "model.gguf"
    previous_artifact.parent.mkdir(parents=True)
    previous_artifact.write_bytes(b"keep-this-valid-candidate")

    with pytest.raises(DeploymentError, match="conversion"):
        deployer.install(valid_candidate_manifest("candidate-v2"), adapter_path)

    assert registry.read().active_model == "baseline-v1"
    assert registry.read().previous_model is None
    assert registry.read().candidates == {}
    assert not list((tmp_path / "candidates").glob(".candidate-v2-gguf.*"))
    assert previous_artifact.read_bytes() == b"keep-this-valid-candidate"
    assert adapter_path.is_dir()
    assert (adapter_path / "adapter-config.json").is_file()


def test_reports_failure_to_clean_incomplete_conversion_output(tmp_path, monkeypatch):
    registry = make_registry(tmp_path)
    deployer = make_deployer(
        registry,
        FakeConverter(failure=RuntimeError("conversion failed")),
    )

    def fail_cleanup(path, **kwargs):
        raise OSError("cleanup denied")

    monkeypatch.setattr(deploy_module.shutil, "rmtree", fail_cleanup)

    with pytest.raises(DeploymentError, match="clean"):
        deployer.install(
            valid_candidate_manifest("candidate-v2"),
            make_adapter(tmp_path),
        )

    assert registry.read().candidates == {}
    assert list((tmp_path / "candidates").glob(".candidate-v2-gguf.*"))


def test_failed_install_preserves_registry_and_does_not_leave_partial_output(tmp_path):
    registry = make_registry(tmp_path)
    converter = FakeConverter()
    ollama = FakeOllamaAdapter(install_error=OllamaError("offline"))
    deployer = make_deployer(registry, converter, ollama)
    adapter_path = make_adapter(tmp_path)

    with pytest.raises(DeploymentError, match="install"):
        deployer.install(valid_candidate_manifest("candidate-v2"), adapter_path)

    state = registry.read()
    assert state.active_model == "baseline-v1"
    assert state.previous_model is None
    assert state.candidates == {}
    assert not list((tmp_path / "candidates").glob(".candidate-v2-gguf.*"))
    assert adapter_path.is_dir()


def test_does_not_register_when_ollama_does_not_confirm_candidate_tag(tmp_path):
    registry = make_registry(tmp_path)
    ollama = FakeOllamaAdapter(confirm_installed=False)
    deployer = make_deployer(registry, ollama=ollama)

    with pytest.raises(DeploymentError, match="installed"):
        deployer.install(
            valid_candidate_manifest("candidate-v2"),
            make_adapter(tmp_path),
        )

    assert registry.read().active_model == "baseline-v1"
    assert registry.read().candidates == {}


def test_rejects_existing_candidate_artifacts_without_deleting_them(tmp_path):
    registry = make_registry(tmp_path)
    deployer = make_deployer(registry)
    existing = tmp_path / "candidates" / "candidate-v2-gguf"
    existing.mkdir(parents=True)
    artifact = existing / "model.gguf"
    artifact.write_bytes(b"previous-valid-artifact")

    with pytest.raises(DeploymentError, match="already exists"):
        deployer.install(
            valid_candidate_manifest("candidate-v2"),
            make_adapter(tmp_path),
        )

    assert artifact.read_bytes() == b"previous-valid-artifact"
    assert registry.read().candidates == {}


def test_refuses_candidate_id_that_collides_with_an_installed_baseline(tmp_path):
    registry = make_registry(tmp_path)
    converter = FakeConverter()
    deployer = make_deployer(registry, converter)

    with pytest.raises(DeploymentError, match="installed|baseline"):
        deployer.install(
            valid_candidate_manifest("baseline-v1"),
            make_adapter(tmp_path, "baseline-v1"),
        )

    assert converter.calls == []
    assert registry.read().active_model == "baseline-v1"


def test_refuses_to_overwrite_an_existing_ollama_tag_before_conversion(tmp_path):
    registry = make_registry(tmp_path)
    converter = FakeConverter()
    ollama = FakeOllamaAdapter()
    ollama.installed_tags.append("candidate-v2")
    deployer = make_deployer(registry, converter, ollama)

    with pytest.raises(DeploymentError, match="already installed"):
        deployer.install(
            valid_candidate_manifest("candidate-v2"),
            make_adapter(tmp_path),
        )

    assert converter.calls == []
    assert registry.read().active_model == "baseline-v1"
    assert registry.read().candidates == {}


def test_rejects_ignored_path_guard_failure_before_conversion(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    candidate_root = repo / "distill" / "models" / "candidates"
    registry = DistillModelRegistry(tmp_path / "registry.json")
    registry.initialize_active("baseline-v1")
    converter = FakeConverter()
    deployer = make_deployer(
        registry,
        converter,
        candidate_root=candidate_root,
    )

    with pytest.raises(DeploymentError, match="gitignored"):
        deployer.install(
            valid_candidate_manifest("candidate-v2"),
            make_adapter(tmp_path),
        )

    assert converter.calls == []
    assert registry.read().candidates == {}
    assert not candidate_root.exists()


def test_candidate_ids_at_schema_limit_receive_safe_output_directory(tmp_path):
    registry = make_registry(tmp_path)
    candidate_id = "c" * 64
    converter = FakeConverter()
    deployer = make_deployer(registry, converter)

    tag = deployer.install(
        valid_candidate_manifest(candidate_id),
        make_adapter(tmp_path, candidate_id),
    )

    assert tag == candidate_id
    artifact_directory = next(
        path for path in (tmp_path / "candidates").iterdir() if path.is_dir()
    )
    assert len(artifact_directory.name) <= 64
    assert artifact_directory.name.endswith("-gguf")
    assert set(registry.read().candidates) == {candidate_id}


def test_does_not_register_when_ollama_returns_different_tag(tmp_path):
    registry = make_registry(tmp_path)
    ollama = FakeOllamaAdapter(returned_tag="other-candidate")
    deployer = make_deployer(registry, ollama=ollama)

    with pytest.raises(DeploymentError, match="different"):
        deployer.install(
            valid_candidate_manifest("candidate-v2"),
            make_adapter(tmp_path),
        )

    assert registry.read().active_model == "baseline-v1"
    assert registry.read().candidates == {}


def test_rejects_converter_output_that_is_not_a_gguf_file(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    wrong_format = stage / "model.bin"
    wrong_format.write_bytes(b"not a GGUF")

    with pytest.raises(DeploymentError, match="GGUF"):
        CandidateDeployer._validate_converted_output(wrong_format, stage)


def test_command_converter_uses_fixed_argv_without_shell_interpolation(tmp_path, monkeypatch):
    base_model = tmp_path / "base model"
    adapter = tmp_path / "adapter model"
    output = tmp_path / "output dir"
    base_model.mkdir()
    adapter.mkdir()
    output.mkdir()
    script = tmp_path / "convert_lora_to_gguf.py"
    script.write_text("# fake converter\n", encoding="utf-8")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        (output / "model-f16.gguf").write_bytes(b"f16")
        (output / "model.gguf").write_bytes(b"quantized")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(deploy_module.subprocess, "run", run)
    converter = SubprocessModelConverter(script, quantizer_executable="llama-quantize")

    result = converter.convert(
        adapter,
        output,
        ConversionConfig(base_model_path=base_model, quantization="Q4_K_M"),
    )

    assert result == output / "model.gguf"
    assert len(calls) == 2
    assert all(call_kwargs["shell"] is False for _, call_kwargs in calls)
    assert all(isinstance(argv, list) for argv, _ in calls)
    assert str(base_model) in calls[0][0]
    assert str(adapter) in calls[0][0]
    assert "Q4_K_M" in calls[1][0]


def test_ollama_adapter_uses_candidate_tag_and_modelfile_argv(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"synthetic")
    calls = []

    def run(argv, **kwargs):
        contents = (
            Path(argv[-1]).read_text(encoding="utf-8")
            if argv[1] == "create"
            else ""
        )
        calls.append((argv, kwargs, contents))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(deploy_module.subprocess, "run", run)
    ollama = SubprocessOllamaAdapter(executable="ollama")

    tag = ollama.install_candidate(model_path, "candidate-v2")

    assert tag == "candidate-v2"
    assert calls[0][0][:3] == ["ollama", "create", "candidate-v2"]
    assert calls[0][1]["shell"] is False
    assert f'FROM "{model_path.resolve()}"' in calls[0][2]
    assert ollama.is_installed(tag) is True
    assert calls[1][0] == ["ollama", "show", "candidate-v2"]


def test_ollama_adapter_rejects_invalid_candidate_tag_before_subprocess(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"synthetic")

    def fail_if_called(*args, **kwargs):
        pytest.fail("invalid candidate tag reached subprocess")

    monkeypatch.setattr(deploy_module.subprocess, "run", fail_if_called)
    ollama = SubprocessOllamaAdapter(executable="ollama")

    with pytest.raises(OllamaError, match="candidate"):
        ollama.install_candidate(model_path, "bad candidate")
