from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import agent.model_client as model_client_module
from agent.config import Settings
from agent.model_client import (
    ModelConfigurationError,
    OpenAICompatibleModelClient,
    resolve_model_name,
)
from agent.models import TaskPlan


class FixedResolver:
    def resolve(self, model_name):
        assert model_name == "tuminh-active"
        return "tuminh:v2"


def make_settings(api_key="test-key", model_name="test-model"):
    import tempfile

    return Settings.from_env(
        {
            "MODEL_API_KEY": api_key,
            "MODEL_NAME": model_name,
            "WORKSPACE_ROOT": tempfile.gettempdir(),
        }
    )


def test_model_name_resolver_preserves_regular_model_names():
    assert resolve_model_name("provider-model", None) == "provider-model"


def test_active_model_sentinel_requires_resolver():
    with pytest.raises(ModelConfigurationError, match="resolver"):
        resolve_model_name("tuminh-active", None)


def test_active_model_sentinel_uses_injected_resolver():
    assert resolve_model_name("tuminh-active", FixedResolver()) == "tuminh:v2"


def test_live_client_requires_api_key():
    with pytest.raises(ModelConfigurationError, match="MODEL_API_KEY"):
        OpenAICompatibleModelClient(make_settings(api_key=""))


def test_client_requests_json_and_validates_response(monkeypatch):
    calls = []

    class FakeOpenAI:
        def __init__(self, *, base_url, api_key):
            assert base_url == "https://api.openai.com/v1"
            assert api_key == "test-key"
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"task_id":"t1","subtasks":[{"id":"s1","description":"Check","files":[],"verification":[]}],"verification_commands":[]}'
                        )
                    )
                ]
            )

    monkeypatch.setattr(model_client_module, "OpenAI", FakeOpenAI)
    client = OpenAICompatibleModelClient(make_settings())

    plan = client.complete_json("policy", '{"task_id":"t1"}', TaskPlan)

    assert isinstance(plan, TaskPlan)
    assert plan.task_id == "t1"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["model"] == "test-model"
    assert calls[0]["messages"] == [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": '{"task_id":"t1"}'},
    ]


def test_client_raises_on_malformed_model_json(monkeypatch):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content='{"task_id":')
                            )
                        ]
                    )
                )
            )

    monkeypatch.setattr(model_client_module, "OpenAI", FakeOpenAI)
    client = OpenAICompatibleModelClient(make_settings())

    with pytest.raises(ValidationError):
        client.complete_json("policy", "payload", TaskPlan)


def test_active_model_name_resolves_at_client_construction(monkeypatch):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: None))

    monkeypatch.setattr(model_client_module, "OpenAI", FakeOpenAI)
    client = OpenAICompatibleModelClient(
        make_settings(model_name="tuminh-active"),
        model_name_resolver=FixedResolver(),
    )

    assert client.model_name == "tuminh:v2"


def test_active_model_alias_is_preserved_in_settings():
    assert make_settings(model_name="tuminh-active").model_name == "tuminh-active"


def test_active_model_name_resolves_through_configured_registry(tmp_path, monkeypatch):
    from distill.registry import DistillModelRegistry

    registry_path = tmp_path / "registry.json"
    DistillModelRegistry(registry_path).initialize_active("baseline-tag")
    settings = Settings.from_env(
        {
            "MODEL_API_KEY": "test-key",
            "MODEL_NAME": "tuminh-active",
            "WORKSPACE_ROOT": str(tmp_path),
            "DISTILL_REGISTRY_PATH": str(registry_path),
        }
    )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(model_client_module, "OpenAI", FakeOpenAI)

    assert OpenAICompatibleModelClient(settings).model_name == "baseline-tag"


def test_active_model_name_missing_registry_fails_before_model_client(
    tmp_path, monkeypatch
):
    settings = Settings.from_env(
        {
            "MODEL_API_KEY": "test-key",
            "MODEL_NAME": "tuminh-active",
            "WORKSPACE_ROOT": str(tmp_path),
        }
    )

    def fail_if_created(**kwargs):
        raise AssertionError("provider client must not be constructed")

    monkeypatch.setattr(model_client_module, "OpenAI", fail_if_created)
    with pytest.raises(ModelConfigurationError, match="active model"):
        OpenAICompatibleModelClient(settings)
