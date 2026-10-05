from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel

from agent.config import Settings
from distill.registry import DistillModelRegistry, RegistryError


TModel = TypeVar("TModel", bound=BaseModel)


class ModelConfigurationError(ValueError):
    """Raised when the live model client cannot be configured safely."""


class ModelResponseError(ValueError):
    """Raised when a provider response does not contain usable JSON content."""


class TransientModelError(RuntimeError):
    """Raised for a provider request failure that is safe to retry."""


class ModelNameResolver(Protocol):
    def resolve(self, model_name: str) -> str: ...


class RegistryModelNameResolver:
    def __init__(self, registry_path: Path):
        self.registry = DistillModelRegistry(registry_path)

    def resolve(self, model_name: str) -> str:
        try:
            return self.registry.resolve_active_model_name(model_name)
        except RegistryError as error:
            raise ModelConfigurationError(
                f"Could not resolve active model from registry: {error}"
            ) from error


class ModelClient(Protocol):
    def complete_json(
        self,
        system_prompt: str,
        user_payload: str,
        response_type: type[TModel],
    ) -> TModel: ...


def resolve_model_name(
    configured_name: str,
    resolver: ModelNameResolver | None,
) -> str:
    if configured_name != "tuminh-active":
        return configured_name
    if resolver is None:
        raise ModelConfigurationError(
            "MODEL_NAME=tuminh-active requires a configured model-name resolver"
        )
    resolved_name = resolver.resolve(configured_name)
    if not resolved_name.strip():
        raise ModelConfigurationError("The active model resolver returned an empty model name")
    return resolved_name


def build_schema_instruction(response_type: type[BaseModel]) -> str:
    """Render an explicit JSON Schema instruction for weaker/local models.

    Generic OpenAI "json_object" mode only guarantees syntactically valid
    JSON, not a specific set of field names. Smaller/local models frequently
    guess wrong field names without an explicit schema, so we inject the
    schema as plain text into the system prompt for every provider.
    """
    schema_json = json.dumps(response_type.model_json_schema(), indent=2)
    return (
        "Respond with a single JSON object that strictly matches the following "
        "JSON Schema. Use exactly these field names, include every required "
        "field, and do not include any field that is not defined in the "
        "schema.\n\nJSON Schema:\n"
        f"{schema_json}"
    )


class OpenAICompatibleModelClient:
    def __init__(
        self,
        settings: Settings,
        model_name_resolver: ModelNameResolver | None = None,
    ):
        if not settings.model_api_key:
            raise ModelConfigurationError(
                "MODEL_API_KEY is required to construct the live model client"
            )
        if settings.model_name == "tuminh-active" and model_name_resolver is None:
            model_name_resolver = RegistryModelNameResolver(
                settings.distill_registry_path
            )
        self.model_name = resolve_model_name(settings.model_name, model_name_resolver)
        self._client = OpenAI(
            base_url=settings.model_base_url,
            api_key=settings.model_api_key,
        )

    def complete_json(
        self,
        system_prompt: str,
        user_payload: str,
        response_type: type[TModel],
    ) -> TModel:
        full_system_prompt = f"{system_prompt}\n\n{build_schema_instruction(response_type)}"
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                response_format={"type": "json_object"},
            )
        except (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        ) as error:
            raise TransientModelError(f"Transient model request failure: {error}") from error
        if not response.choices or not response.choices[0].message.content:
            raise ModelResponseError("Model response did not contain JSON content")
        return response_type.model_validate_json(response.choices[0].message.content)
