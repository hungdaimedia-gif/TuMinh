from pathlib import Path

import pytest

from agent.config import Settings, SettingsError


def test_settings_read_openai_compatible_endpoint(tmp_path: Path) -> None:
    settings = Settings.from_env(
        {
            "MODEL_API_KEY": "test-key",
            "MODEL_BASE_URL": "https://model.example/v1",
            "MODEL_NAME": "test-model",
            "WORKSPACE_ROOT": str(tmp_path),
        }
    )

    assert settings.model_base_url == "https://model.example/v1"
    assert settings.model_name == "test-model"
    assert settings.workspace_root == tmp_path.resolve()
    assert settings.max_steps > 0
    assert settings.task_source == "local"


@pytest.mark.parametrize("name", ["MAX_STEPS", "MAX_POLL_CYCLES", "COMMAND_TIMEOUT_SECONDS"])
def test_settings_reject_nonpositive_limits(name: str) -> None:
    with pytest.raises(SettingsError):
        Settings.from_env({name: "0"})


def test_settings_reject_unknown_task_source() -> None:
    with pytest.raises(SettingsError):
        Settings.from_env({"TASK_SOURCE": "somewhere-else"})


@pytest.mark.parametrize("name", ["MODEL_BASE_URL", "MODEL_NAME"])
def test_settings_reject_empty_model_identity(name: str) -> None:
    with pytest.raises(SettingsError):
        Settings.from_env({name: "   "})


def test_settings_do_not_require_model_key_for_offline_components() -> None:
    settings = Settings.from_env({})

    assert settings.model_api_key is None


def test_pull_request_creation_is_opt_in(tmp_path: Path) -> None:
    default_settings = Settings.from_env({"WORKSPACE_ROOT": str(tmp_path)})
    enabled_settings = Settings.from_env(
        {
            "WORKSPACE_ROOT": str(tmp_path),
            "CREATE_PULL_REQUEST": "true",
        }
    )

    assert default_settings.create_pull_request is False
    assert enabled_settings.create_pull_request is True


def test_settings_reject_invalid_pull_request_flag(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="CREATE_PULL_REQUEST"):
        Settings.from_env(
            {
                "WORKSPACE_ROOT": str(tmp_path),
                "CREATE_PULL_REQUEST": "sometimes",
            }
        )
