"""Provider interface shared by the live adapters and the mock."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from tilebench.llm.v2.providers.usage import NormalizedUsage


class TransportError(RuntimeError):
    """Network/transport failure. Usage may be unknown; charging is uncertain."""


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

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usage"] = self.usage.to_dict()
        return d


class Provider(Protocol):
    name: str
    usage_schema: str

    def generate(self, request: GenerationRequest) -> GenerationResult: ...
