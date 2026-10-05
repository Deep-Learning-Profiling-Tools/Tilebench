"""Deterministic scripted provider for tests and dry runs. Never touches the
network. Each script item is consumed by one generate() call:

    {"text": "...", "usage": {...raw usage in the chosen schema...},
     "transport_failures": 0, "usage_missing": False}

`transport_failures` > 0 raises TransportError that many times before
succeeding (consumes one script item per success, the runner counts the
failed transport attempts separately). `usage_missing` returns a response
whose usage is unknown (simulates a timeout after the request was accepted)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from tilebench.llm.v2.providers.base import GenerationRequest, GenerationResult, TransportError
from tilebench.llm.v2.providers.usage import ANTHROPIC_SCHEMA, OPENAI_SCHEMA, normalize, unknown


@dataclass
class MockProvider:
    script: list[dict]
    usage_schema: str = OPENAI_SCHEMA
    name: str = "mock"
    requests: list[GenerationRequest] = field(default_factory=list)
    _pending_failures: int = 0
    _cursor: int = 0
    _armed: int = -1

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if self._cursor >= len(self.script):
            raise RuntimeError("mock provider script exhausted")
        item = self.script[self._cursor]
        if self._armed != self._cursor:               # arm the failures of this item exactly once
            self._pending_failures = int(item.get("transport_failures", 0))
            self._armed = self._cursor
        if self._pending_failures > 0:
            self._pending_failures -= 1
            raise TransportError(f"mock transport failure for script item {self._cursor}",
                                 charged=item.get("transport_charged", "no"))
        self._cursor += 1
        if item.get("usage_missing"):
            usage = unknown(self.name, self.usage_schema, "mock: usage withheld")
            raw = None
        else:
            raw = item.get("usage") or _default_usage(self.usage_schema, request, item.get("text", ""))
            usage = normalize(self.usage_schema, raw)
        return GenerationResult(text=item.get("text"), model_id=request.model_id, provider=self.name,
                                response_id=f"mock-{self._cursor}", usage_raw=raw, usage=usage,
                                transport_attempts=1, elapsed_s=0.0, raw_response={"mock": True},
                                error=item.get("error"))


def _default_usage(schema: str, req: GenerationRequest, text: str) -> dict:
    n_in = (len(req.system) + len(req.user)) // 4 + 1
    n_out = len(text) // 4 + 1
    if schema == OPENAI_SCHEMA:
        return {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out,
                "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}
    if schema == ANTHROPIC_SCHEMA:
        return {"input_tokens": n_in, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                "output_tokens": n_out}
    raise ValueError(schema)


def scripted_text(filename: str, body: str) -> str:
    """A response in the exact format the strict parser accepts."""
    return f"```python title=\"{filename}\"\n{body}\n```\n"


def timestamp() -> float:
    return time.time()
