"""Anthropic Messages API adapter (streaming, lazy SDK import, no internal retries).

Request shape (checked against anthropic 0.103.1 on 2026-10-05):
    client.messages.stream(model=..., max_tokens=..., system=..., messages=[user],
                           thinking={"type": "adaptive"}, output_config={"effort": "xhigh"})
The final Message carries `stop_reason` (end_turn | max_tokens | stop_sequence |
refusal | ...), `usage` (input/output/cache fields), `content` (text and
thinking blocks), `id` and `model`.

The API key is read from the environment variable named by `api_key_env`
(the study's servers export CLAUDE_API_KEY; ANTHROPIC_API_KEY is not
assumed). The key never leaves this process: it is not logged, not written
into any manifest, request archive or response, and the evaluation worker
does not inherit it.

Error classification:
    400/401/403/404/422                       -> ProviderConfigError (never retried)
    APIConnectionError (not timeout), 429, 5xx -> TransportError(charged="no")
    APITimeoutError, stream interrupted       -> TransportError(charged="unknown")
On a stream interruption the last usage numbers seen (message_start /
message_delta) are attached as `usage_partial`, a lower bound only.
"""
from __future__ import annotations

import os
import time
from typing import Any

from tilebench.llm.v2.providers.base import (GenerationRequest, GenerationResult, ProviderConfigError, TransportError,
                                             response_headers, safe_error_body)
from tilebench.llm.v2.providers.usage import ANTHROPIC_SCHEMA, normalize, unknown

DEFAULT_API_KEY_ENV = "CLAUDE_API_KEY"
_CONFIG_STATUS = {400, 401, 403, 404, 422}


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def classify_exception(e: Exception) -> Exception:
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return TransportError(f"{type(e).__name__}: {e}", charged="unknown")
    if isinstance(e, anthropic.APITimeoutError):
        return TransportError(f"APITimeoutError: {e}", charged="unknown")
    if isinstance(e, anthropic.APIConnectionError):
        return TransportError(f"APIConnectionError: {e}", charged="no")
    if isinstance(e, anthropic.APIStatusError):
        code = getattr(e, "status_code", None)
        if code in _CONFIG_STATUS:
            return ProviderConfigError(f"{type(e).__name__} ({code}): {e}", status_code=code,
                                       kind="auth" if code in (401, 403) else "request")
        return TransportError(f"{type(e).__name__} ({code}): {e}", charged="no")
    return TransportError(f"{type(e).__name__}: {e}", charged="unknown")


def build_kwargs(request: GenerationRequest) -> dict[str, Any]:
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
        kwargs["thinking"] = dict(s["thinking"])
    if s.get("output_effort"):
        kwargs["output_config"] = {"effort": s["output_effort"]}
    if s.get("temperature") is not None:
        kwargs["temperature"] = float(s["temperature"])
    return kwargs


def _usage_from_event(event: Any, partial: dict | None) -> dict | None:
    """Track usage announced on the stream (message_start carries input
    counts, message_delta carries cumulative output counts)."""
    et = getattr(event, "type", "")
    if et == "message_start":
        msg = getattr(event, "message", None)
        u = _dump(getattr(msg, "usage", None)) if msg is not None else None
        return dict(u) if isinstance(u, dict) else partial
    if et == "message_delta":
        u = _dump(getattr(event, "usage", None))
        if isinstance(u, dict):
            merged = dict(partial or {})
            merged.update({k: v for k, v in u.items() if v is not None})
            return merged
    return partial


class AnthropicMessagesProvider:
    name = "anthropic"
    usage_schema = ANTHROPIC_SCHEMA

    def __init__(self, *, timeout_s: float = 3600.0, api_key_env: str = DEFAULT_API_KEY_ENV):
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} is not set; live Anthropic calls are blocked")
        from anthropic import Anthropic  # lazy
        self.api_key_env = api_key_env
        self._client = Anthropic(api_key=key, timeout=timeout_s, max_retries=0)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        t0 = time.time()
        kwargs = build_kwargs(request)
        sent = {k: v for k, v in kwargs.items() if k not in ("system", "messages")}
        events = 0
        text_chars = 0
        partial: dict | None = None
        msg = None
        headers = None
        last_event = None
        try:
            with self._client.messages.stream(**kwargs) as stream:
                headers = response_headers(stream)
                for event in stream:
                    events += 1
                    last_event = getattr(event, "type", "") or last_event
                    partial = _usage_from_event(event, partial)
                    if getattr(event, "type", "") == "text":
                        text_chars += len(getattr(event, "text", "") or "")
                msg = stream.get_final_message()
        except (TransportError, ProviderConfigError):
            raise
        except Exception as e:  # noqa: BLE001
            err = classify_exception(e)
            rl = headers or response_headers(e)
            if isinstance(err, TransportError):
                if events > 0:
                    err = TransportError(str(err), charged="unknown", usage_partial=partial, stream_events=events,
                                         partial_text_chars=text_chars)
                err.rate_limit, err.last_event = rl, last_event
                err.error_body = safe_error_body(getattr(e, "body", None))
                err.status_code = getattr(e, "status_code", None)
            elif isinstance(err, ProviderConfigError):
                err.rate_limit = rl
            raise err from e
        raw = _dump(msg)
        usage_raw = raw.get("usage") if isinstance(raw, dict) else None
        usage = normalize(ANTHROPIC_SCHEMA, usage_raw, raw if isinstance(raw, dict) else None) if usage_raw else unknown(self.name, ANTHROPIC_SCHEMA, "usage absent")
        stop = raw.get("stop_reason") if isinstance(raw, dict) else None
        truncated = stop == "max_tokens"
        error = None
        if stop == "max_tokens":
            error = "stop_reason=max_tokens: output truncated at the configured cap"
        elif stop == "refusal":
            error = "stop_reason=refusal"
        text = "".join(block.get("text", "") for block in raw.get("content", []) if block.get("type") == "text") \
            if isinstance(raw, dict) else None
        return GenerationResult(text=text, model_id=raw.get("model", request.model_id) if isinstance(raw, dict) else request.model_id,
                                provider=self.name, response_id=raw.get("id") if isinstance(raw, dict) else None,
                                usage_raw=usage_raw, usage=usage, transport_attempts=1, elapsed_s=time.time() - t0,
                                error=error, raw_response=raw, terminal_status=stop, truncated=truncated,
                                stream_events=events, streamed=True, requested_model_id=request.model_id,
                                request_settings_sent=sent, rate_limit=headers)
