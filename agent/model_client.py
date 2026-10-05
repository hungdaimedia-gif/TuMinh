from __future__ import annotations

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


TModel = TypeVar("TModel", bound=BaseModel)


class ModelConfigurationError(ValueError):
    """Raised when the live model client cannot be configured safely."""


class ModelResponseError(ValueError):
    """Raised when a provider response does not contain usable JSON content."""


class TransientModelError(RuntimeError):
    """Raised for a provider request failure that is safe to retry."""


class ModelNameResolver(Protocol):
    def resolve(self, model_name: str) -> str: ...


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
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
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
