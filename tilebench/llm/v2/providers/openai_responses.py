"""OpenAI Responses API adapter. The SDK is imported lazily; the module can be
imported and unit-tested without the `openai` package or credentials.

No retry logic lives here: the orchestration layer counts transport attempts
and decides whether to re-issue a request, so that every attempt is logged."""
from __future__ import annotations

import os
import time
from typing import Any

from tilebench.llm.v2.providers.base import GenerationRequest, GenerationResult, TransportError
from tilebench.llm.v2.providers.usage import OPENAI_SCHEMA, normalize, unknown


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


class OpenAIResponsesProvider:
    name = "openai"
    usage_schema = OPENAI_SCHEMA

    def __init__(self, *, timeout_s: float = 1800.0, api_key_env: str = "OPENAI_API_KEY"):
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} is not set; live OpenAI calls are blocked")
        from openai import OpenAI  # lazy
        self._client = OpenAI(api_key=key, timeout=timeout_s, max_retries=0)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        t0 = time.time()
        kwargs: dict[str, Any] = {
            "model": request.model_id,
            "input": [{"role": "system", "content": request.system},
                      {"role": "user", "content": request.user}],
        }
        s = request.settings or {}
        if s.get("reasoning_effort"):
            kwargs["reasoning"] = {"effort": s["reasoning_effort"]}
        if s.get("max_output_tokens"):
            kwargs["max_output_tokens"] = int(s["max_output_tokens"])
        try:
            resp = self._client.responses.create(**kwargs)
        except Exception as e:  # SDK exception types vary; the caller classifies by message
            raise TransportError(f"{type(e).__name__}: {e}") from e
        raw = _dump(resp)
        usage_raw = raw.get("usage") if isinstance(raw, dict) else None
        usage = normalize(OPENAI_SCHEMA, usage_raw) if usage_raw else unknown(self.name, OPENAI_SCHEMA, "usage absent")
        text = getattr(resp, "output_text", None)
        return GenerationResult(text=text, model_id=getattr(resp, "model", request.model_id), provider=self.name,
                                response_id=getattr(resp, "id", None), usage_raw=usage_raw, usage=usage,
                                transport_attempts=1, elapsed_s=time.time() - t0, raw_response=raw)
