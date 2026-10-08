"""Provider interface shared by the live adapters and the mock.

Error classes (the orchestration layer decides what to do with each):

- TransportError      a network/transport failure. `charged` says whether the
                      provider can have billed the request: "no" when the
                      request never reached the provider or was rejected with
                      an error status before any generation (connection
                      refused, HTTP 429/5xx), "unknown" when it may have been
                      processed (timeout after sending, stream interrupted
                      after events arrived). `usage_partial` carries the last
                      usage numbers seen on the stream, a LOWER BOUND only.
- ProviderConfigError authentication, permission, invalid model/parameter
                      (HTTP 400/401/403/404/422). Re-sending the same request
                      cannot succeed; it is never retried and the trajectory
                      stops with the provider's message recorded verbatim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from tilebench.llm.v2.providers.usage import NormalizedUsage

CHARGE_STATES = ("no", "unknown", "known")

# Non-sensitive response headers archived for scheduling analysis (rate-limit
# state, retry hints, request ids). Whitelist only: authorization, cookies,
# organization/project identifiers and every other header are never kept.
# They are telemetry: never part of a prompt, a feedback text or a score.
RATE_LIMIT_HEADER_PREFIXES = ("x-ratelimit-", "anthropic-ratelimit-", "anthropic-priority-")
RATE_LIMIT_HEADER_NAMES = ("retry-after", "retry-after-ms", "x-request-id", "request-id", "date")
_SENSITIVE_HEADER_PARTS = ("authorization", "api-key", "cookie", "organization", "project", "token-id", "x-api-key")


def rate_limit_headers(headers: Any) -> dict | None:
    """Whitelisted, lower-cased copy of the rate-limit/request-id response
    headers; None when no headers are available."""
    if headers is None:
        return None
    try:
        items = list(headers.items())
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for k, v in items:
        lk = str(k).lower()
        if any(part in lk for part in _SENSITIVE_HEADER_PARTS) and not lk.startswith(RATE_LIMIT_HEADER_PREFIXES):
            continue
        if lk.startswith(RATE_LIMIT_HEADER_PREFIXES) or lk in RATE_LIMIT_HEADER_NAMES:
            out[lk] = str(v)
    return out


def response_headers(obj: Any) -> dict | None:
    """Rate-limit headers of an httpx.Response, or of an object carrying one
    in `.response` / `._response` (SDK stream objects, APIStatusError)."""
    for cand in (obj, getattr(obj, "response", None), getattr(obj, "_response", None)):
        h = getattr(cand, "headers", None)
        if h is not None:
            return rate_limit_headers(h)
    return None


def _duration_s(text: str) -> float | None:
    """'1s', '6m0s', '20ms', '1h2m3.5s', '12' (seconds) -> seconds."""
    import re
    t = str(text).strip()
    if re.fullmatch(r"\d+(\.\d+)?", t):
        return float(t)
    total, matched = 0.0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", t):
        matched = True
        total += float(num) * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return total if matched else None


def retry_after_seconds(headers: dict | None, now: float | None = None) -> float:
    """Seconds to wait before the next request according to the provider's
    headers. Retry-After / retry-after-ms win; otherwise the reset time of
    every limit whose remaining count is 0 (OpenAI x-ratelimit-reset-* are
    durations, Anthropic anthropic-ratelimit-*-reset are RFC 3339 times).
    0 when nothing applies (the caller's fixed backoff is used)."""
    import time
    from datetime import datetime
    if not headers:
        return 0.0
    now = time.time() if now is None else now
    if "retry-after-ms" in headers:
        try:
            return float(headers["retry-after-ms"]) / 1000.0
        except ValueError:
            pass
    if "retry-after" in headers:
        d = _duration_s(headers["retry-after"])
        if d is not None:
            return d
    waits = []
    for k, v in headers.items():
        if k.startswith("x-ratelimit-remaining-") and str(v).strip() == "0":
            d = _duration_s(headers.get("x-ratelimit-reset-" + k[len("x-ratelimit-remaining-"):], ""))
            if d is not None:
                waits.append(d)
        elif k.startswith(("anthropic-ratelimit-", "anthropic-priority-")) and k.endswith("-remaining") and str(v).strip() == "0":
            reset = headers.get(k[: -len("-remaining")] + "-reset")
            if reset:
                try:
                    waits.append(max(0.0, datetime.fromisoformat(str(reset).replace("Z", "+00:00")).timestamp() - now))
                except ValueError:
                    pass
    return max(waits) if waits else 0.0


def safe_error_body(body: Any, limit: int = 2000) -> Any:
    """The provider's error object (type/code/message), JSON-able and bounded."""
    if body is None:
        return None
    try:
        import json
        text = json.dumps(body, default=str)
    except Exception:  # noqa: BLE001
        text = str(body)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


class TransportError(RuntimeError):
    def __init__(self, message: str, *, charged: str = "unknown", usage_partial: dict | None = None,
                 stream_events: int = 0, partial_text_chars: int = 0, rate_limit: dict | None = None,
                 last_event: str | None = None, error_body: Any = None, status_code: int | None = None):
        super().__init__(message)
        if charged not in ("no", "unknown"):
            raise ValueError(charged)
        self.charged = charged
        self.usage_partial = usage_partial
        self.stream_events = stream_events
        self.partial_text_chars = partial_text_chars
        self.rate_limit = rate_limit
        self.last_event = last_event
        self.error_body = error_body
        self.status_code = status_code

    def record(self) -> dict:
        return {"error": str(self), "charged": self.charged, "usage_partial": self.usage_partial,
                "stream_events": self.stream_events, "partial_text_chars": self.partial_text_chars,
                "rate_limit": self.rate_limit, "last_event": self.last_event, "error_body": self.error_body,
                "status_code": self.status_code}


class StopRequested(RuntimeError):
    """A scheduler pause was requested before this request was sent: nothing
    was sent, nothing is charged, the trajectory pauses at a resumable boundary."""


class ProviderConfigError(RuntimeError):
    """Non-retryable provider rejection (auth, permission, model, parameters)."""

    def __init__(self, message: str, *, status_code: int | None = None, kind: str = "config",
                 rate_limit: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.kind = kind
        self.rate_limit = rate_limit

    def record(self) -> dict:
        return {"error": str(self), "status_code": self.status_code, "kind": self.kind, "charged": "no",
                "rate_limit": self.rate_limit}


@dataclass
class GenerationRequest:
    system: str
    user: str
    model_id: str
    provider: str                     # openai | anthropic | mock
    settings: dict = field(default_factory=dict)   # reasoning/thinking/max_output_tokens as configured
    metadata: dict = field(default_factory=dict)   # task/round/attempt identity (never sent)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationResult:
    text: str | None
    model_id: str
    provider: str
    response_id: str | None
    usage_raw: dict | None
    usage: NormalizedUsage
    transport_attempts: int
    elapsed_s: float
    error: str | None = None          # set when no response was obtained
    raw_response: Any = None          # JSON-able dump of the provider response
    terminal_status: str | None = None   # provider's own terminal state (completed / end_turn / max_tokens / ...)
    truncated: bool = False           # output hit max_output_tokens: not a complete candidate
    stream_events: int = 0            # events received on the stream
    streamed: bool = False
    requested_model_id: str | None = None   # what was asked for (model_id is what the provider echoed)
    request_settings_sent: dict | None = None   # the provider-side parameters actually sent (no secrets)
    rate_limit: dict | None = None    # whitelisted rate-limit / request-id response headers (telemetry only)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usage"] = self.usage.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationResult":
        u = d["usage"]
        usage = u if isinstance(u, NormalizedUsage) else NormalizedUsage(**u)
        return cls(**{k: v for k, v in d.items() if k != "usage"}, usage=usage)


class Provider(Protocol):
    name: str
    usage_schema: str

    def generate(self, request: GenerationRequest) -> GenerationResult: ...
