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


class TransportError(RuntimeError):
    def __init__(self, message: str, *, charged: str = "unknown", usage_partial: dict | None = None,
                 stream_events: int = 0, partial_text_chars: int = 0):
        super().__init__(message)
        if charged not in ("no", "unknown"):
            raise ValueError(charged)
        self.charged = charged
        self.usage_partial = usage_partial
        self.stream_events = stream_events
        self.partial_text_chars = partial_text_chars

    def record(self) -> dict:
        return {"error": str(self), "charged": self.charged, "usage_partial": self.usage_partial,
                "stream_events": self.stream_events, "partial_text_chars": self.partial_text_chars}


class ProviderConfigError(RuntimeError):
    """Non-retryable provider rejection (auth, permission, model, parameters)."""

    def __init__(self, message: str, *, status_code: int | None = None, kind: str = "config"):
        super().__init__(message)
        self.status_code = status_code
        self.kind = kind

    def record(self) -> dict:
        return {"error": str(self), "status_code": self.status_code, "kind": self.kind, "charged": "no"}


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
