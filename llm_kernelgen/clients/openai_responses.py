"""OpenAI Responses API client.

Preferred for OpenAI official endpoints and reasoning models.
Uses ``client.responses.create`` which supports the ``reasoning`` parameter.

Reference:
    https://platform.openai.com/docs/guides/text?api-mode=responses
"""

from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

from llm_kernelgen.clients.base import BaseClient, LLMResponse


class OpenAIResponsesClient(BaseClient):
    """Client that targets the OpenAI Responses API (``/v1/responses``)."""

    def __init__(
        self,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        api_key_env:
            Name of the environment variable that holds the API key.
        base_url:
            Optional base URL override (e.g. for self-hosted endpoints).
        """
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{api_key_env}' is not set or is empty."
            )
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def generate(
        self,
        *,
        model: str,
        instructions: str,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 12000,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion via the Responses API.

        Parameters
        ----------
        model:
            Model identifier (e.g. ``"gpt-4o"``, ``"o3"``).
        instructions:
            System prompt / instructions text.
        prompt:
            User message content.
        temperature:
            Sampling temperature (0 – 2).
        max_output_tokens:
            Maximum tokens in the response.
        reasoning_effort:
            Reasoning effort for o-series models: ``"low"``, ``"medium"``,
            ``"high"``.  Ignored when the model does not support it.
        **kwargs:
            Forwarded verbatim to ``responses.create``.
        """
        req: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if reasoning_effort is not None:
            req["reasoning"] = {"effort": reasoning_effort}
        req.update(kwargs)

        t0 = time.monotonic()
        resp = self.client.responses.create(**req)
        latency = time.monotonic() - t0

        # Extract token usage when available.
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        completion_tokens = getattr(usage, "output_tokens", 0) or 0
        reasoning_tokens = 0
        if usage is not None:
            details = getattr(usage, "output_tokens_details", None)
            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

        return LLMResponse(
            text=resp.output_text,
            raw=resp,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_s=latency,
            model_used=getattr(resp, "model", model),
        )
