"""OpenAI Responses API adapter (streaming).

The SDK is imported lazily; the module can be imported and unit-tested
without the `openai` package or credentials.

No retry logic lives here: the orchestration layer counts transport attempts
and decides whether to re-issue a request, so that every attempt is logged.

Request shape (Responses API, checked against openai 2.37.0 on 2026-10-05):
    client.responses.stream(model=..., input=[system, user], reasoning={"effort": ...},
                            max_output_tokens=...)
The stream is consumed to its end; the final Response object carries
`status` (completed | incomplete | failed), `incomplete_details.reason`
(e.g. max_output_tokens), `usage`, `output_text`, `id` and `model`.

Error classification:
    400/401/403/404/422 (BadRequest, Authentication, PermissionDenied,
    NotFound, UnprocessableEntity)        -> ProviderConfigError (never retried)
    APIConnectionError (not timeout), 429, 5xx -> TransportError(charged="no")
    APITimeoutError, stream interrupted      -> TransportError(charged="unknown")
"""
from __future__ import annotations

import os
import time
from typing import Any

from tilebench.llm.v2.providers.base import GenerationRequest, GenerationResult, ProviderConfigError, TransportError
from tilebench.llm.v2.providers.usage import OPENAI_SCHEMA, normalize, unknown

DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
_CONFIG_STATUS = {400, 401, 403, 404, 422}


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def classify_exception(e: Exception) -> Exception:
    """Map an SDK exception to ProviderConfigError / TransportError."""
    try:
        import openai
    except ImportError:  # pragma: no cover - the SDK is present wherever this is called live
        return TransportError(f"{type(e).__name__}: {e}", charged="unknown")
    if isinstance(e, openai.APITimeoutError):
        return TransportError(f"APITimeoutError: {e}", charged="unknown")
    if isinstance(e, openai.APIConnectionError):
        return TransportError(f"APIConnectionError: {e}", charged="no")
    if isinstance(e, openai.APIStatusError):
        code = getattr(e, "status_code", None)
        if code in _CONFIG_STATUS:
            return ProviderConfigError(f"{type(e).__name__} ({code}): {e}", status_code=code,
                                       kind="auth" if code in (401, 403) else "request")
        return TransportError(f"{type(e).__name__} ({code}): {e}", charged="no")
    return TransportError(f"{type(e).__name__}: {e}", charged="unknown")


def build_kwargs(request: GenerationRequest) -> dict[str, Any]:
    s = request.settings or {}
    kwargs: dict[str, Any] = {
        "model": request.model_id,
        "input": [{"role": "system", "content": request.system},
                  {"role": "user", "content": request.user}],
    }
    if not s.get("reasoning_effort"):
        raise ValueError("openai adapter requires settings.reasoning_effort")
    if not s.get("max_output_tokens"):
        raise ValueError("openai adapter requires settings.max_output_tokens")
    kwargs["reasoning"] = {"effort": s["reasoning_effort"]}
    kwargs["max_output_tokens"] = int(s["max_output_tokens"])
    if s.get("temperature") is not None:
        kwargs["temperature"] = float(s["temperature"])
    if s.get("store") is not None:
        kwargs["store"] = bool(s["store"])
    return kwargs


class OpenAIResponsesProvider:
    name = "openai"
    usage_schema = OPENAI_SCHEMA

    def __init__(self, *, timeout_s: float = 3600.0, api_key_env: str = DEFAULT_API_KEY_ENV):
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} is not set; live OpenAI calls are blocked")
        from openai import OpenAI  # lazy
        self.api_key_env = api_key_env
        self._client = OpenAI(api_key=key, timeout=timeout_s, max_retries=0)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        t0 = time.time()
        kwargs = build_kwargs(request)
        sent = {k: v for k, v in kwargs.items() if k != "input"}
        events = 0
        text_chars = 0
        final = None
        try:
            with self._client.responses.stream(**kwargs) as stream:
                for event in stream:
                    events += 1
                    et = getattr(event, "type", "")
                    if et == "response.output_text.delta":
                        text_chars += len(getattr(event, "delta", "") or "")
                    elif et == "error":
                        raise TransportError(f"stream error event: {_dump(event)}", charged="unknown",
                                             stream_events=events, partial_text_chars=text_chars)
                final = stream.get_final_response()
        except (TransportError, ProviderConfigError):
            raise
        except Exception as e:  # noqa: BLE001 - classified below
            err = classify_exception(e)
            if isinstance(err, TransportError):
                if events > 0:
                    # the provider already produced events: generation may be billed
                    err = TransportError(str(err), charged="unknown", stream_events=events, partial_text_chars=text_chars)
            raise err from e
        raw = _dump(final)
        usage_raw = raw.get("usage") if isinstance(raw, dict) else None
        usage = normalize(OPENAI_SCHEMA, usage_raw) if usage_raw else unknown(self.name, OPENAI_SCHEMA, "usage absent")
        status = raw.get("status") if isinstance(raw, dict) else None
        incomplete = raw.get("incomplete_details") if isinstance(raw, dict) else None
        reason = (incomplete or {}).get("reason") if isinstance(incomplete, dict) else None
        terminal = status if status else None
        if reason:
            terminal = f"{status}:{reason}"
        truncated = status == "incomplete" and reason == "max_output_tokens"
        error = None
        if status == "failed":
            error = f"response.failed: {raw.get('error')}"
        elif status == "incomplete":
            error = f"response.incomplete: {reason}"
        text = getattr(final, "output_text", None)
        return GenerationResult(text=text, model_id=getattr(final, "model", request.model_id), provider=self.name,
                                response_id=getattr(final, "id", None), usage_raw=usage_raw, usage=usage,
                                transport_attempts=1, elapsed_s=time.time() - t0, error=error, raw_response=raw,
                                terminal_status=terminal, truncated=truncated, stream_events=events, streamed=True,
                                requested_model_id=request.model_id, request_settings_sent=sent)
