"""OpenAI Chat Completions compatible client.

Targets any endpoint that implements the ``/chat/completions`` interface,
including OpenAI itself, OpenRouter, and self-hosted providers such as
vLLM, Ollama, or vendor proxies.

The ``extra_body`` parameter lets you transparently forward provider-specific
fields (e.g. ``{"reasoning": {"effort": "medium"}}`` for models that accept
a thinking budget via a non-standard field).
"""

from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

from llm_kernelgen.clients.base import BaseClient, LLMResponse


class OpenAIChatCompatClient(BaseClient):
    """Client that targets any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        api_key_env: str = "OPENAI_COMPAT_API_KEY",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        """
        Parameters
        ----------
        api_key_env:
            Environment variable that stores the API key.
        base_url:
            Base URL of the provider (default: OpenAI official).
        """
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{api_key_env}' is not set or is empty."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        *,
        model: str,
        instructions: str,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 12000,
        reasoning_effort: str | None = None,
        extra_body: dict | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion via the chat completions endpoint.

        Parameters
        ----------
        model:
            Model identifier string.
        instructions:
            System prompt text (mapped to ``{"role": "system", ...}``).
        prompt:
            User turn content.
        temperature:
            Sampling temperature.
        max_output_tokens:
            Mapped to ``max_tokens`` in the API request.
        reasoning_effort:
            If provided and ``extra_body`` does not already contain a
            ``reasoning`` key, injects ``{"reasoning": {"effort": value}}``
            into ``extra_body`` for providers that accept it.
        extra_body:
            Arbitrary additional fields forwarded in the request body.
            Useful for provider-specific extensions.
        **kwargs:
            Additional parameters forwarded to ``chat.completions.create``.
        """
        body: dict[str, Any] = extra_body.copy() if extra_body else {}
        if reasoning_effort is not None and "reasoning" not in body:
            body["reasoning"] = {"effort": reasoning_effort}

        t0 = time.monotonic()
        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
            extra_body=body or None,
            **kwargs,
        )
        latency = time.monotonic() - t0

        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        return LLMResponse(
            text=text,
            raw=resp,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency,
            model_used=getattr(resp, "model", model),
        )
