"""Anthropic Messages API adapter (lazy SDK import, no internal retries)."""
from __future__ import annotations

import os
import time
from typing import Any

from tilebench.llm.v2.providers.base import GenerationRequest, GenerationResult, TransportError
from tilebench.llm.v2.providers.usage import ANTHROPIC_SCHEMA, normalize, unknown


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


class AnthropicMessagesProvider:
    name = "anthropic"
    usage_schema = ANTHROPIC_SCHEMA

    def __init__(self, *, timeout_s: float = 1800.0, api_key_env: str = "ANTHROPIC_API_KEY"):
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} is not set; live Anthropic calls are blocked")
        from anthropic import Anthropic  # lazy
        self._client = Anthropic(api_key=key, timeout=timeout_s, max_retries=0)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        t0 = time.time()
        s = request.settings or {}
        if not s.get("max_output_tokens"):
            raise ValueError("anthropic adapter requires settings.max_output_tokens")
        kwargs: dict[str, Any] = {
            "model": request.model_id,
            "max_tokens": int(s["max_output_tokens"]),
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
        }
        if s.get("thinking"):
            kwargs["thinking"] = s["thinking"]
        if s.get("output_effort"):
            kwargs["output_config"] = {"effort": s["output_effort"]}
        try:
            msg = self._client.messages.create(**kwargs)
        except Exception as e:
            raise TransportError(f"{type(e).__name__}: {e}") from e
        raw = _dump(msg)
        usage_raw = raw.get("usage") if isinstance(raw, dict) else None
        usage = normalize(ANTHROPIC_SCHEMA, usage_raw) if usage_raw else unknown(self.name, ANTHROPIC_SCHEMA, "usage absent")
        text = "".join(block.get("text", "") for block in raw.get("content", []) if block.get("type") == "text") \
            if isinstance(raw, dict) else None
        return GenerationResult(text=text, model_id=raw.get("model", request.model_id) if isinstance(raw, dict) else request.model_id,
                                provider=self.name, response_id=raw.get("id") if isinstance(raw, dict) else None,
                                usage_raw=usage_raw, usage=usage, transport_attempts=1,
                                elapsed_s=time.time() - t0, raw_response=raw)
