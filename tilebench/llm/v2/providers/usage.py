"""Provider usage normalization.

Logical token axis used by the cost metric C_i (study §3.4):

OpenAI Responses API  (schema openai_responses_v1)
    usage.input_tokens           total prompt tokens INCLUDING cached ones
    usage.output_tokens          total completion tokens INCLUDING reasoning
    usage.input_tokens_details.cached_tokens      subset of input_tokens
    usage.output_tokens_details.reasoning_tokens  subset of output_tokens
    usage.total_tokens           = input_tokens + output_tokens
  Source: OpenAI API reference, "Responses" object, `usage`
  (https://platform.openai.com/docs/api-reference/responses/object), checked 2026-10-05.

Anthropic Messages API (schema anthropic_messages_v1)
    usage.input_tokens                 UNCACHED prompt tokens only
    usage.cache_creation_input_tokens  tokens written to the prompt cache
    usage.cache_read_input_tokens      tokens served from the prompt cache
    usage.output_tokens                completion tokens (thinking included)
    usage.cache_creation.{ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}
                                       breakdown of cache_creation_input_tokens (not added again)
  logical input = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
  Source: Anthropic API reference, Messages "usage"
  (https://docs.anthropic.com/en/api/messages), checked 2026-10-05.

A missing field is reported as None and the record status becomes "partial"
or "unknown"; nothing is imputed as 0, and a missing reasoning count is not
interpreted as "the model did not reason"."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

OPENAI_SCHEMA = "openai_responses_v1"
ANTHROPIC_SCHEMA = "anthropic_messages_v1"
CHECKED_AT = "2026-10-05"
DOC_SOURCES = {
    OPENAI_SCHEMA: "https://platform.openai.com/docs/api-reference/responses/object#usage",
    ANTHROPIC_SCHEMA: "https://docs.anthropic.com/en/api/messages#response-usage",
}


@dataclass
class NormalizedUsage:
    provider: str
    schema: str
    status: str                       # ok | partial | unknown
    logical_input: int | None
    logical_output: int | None
    logical_total: int | None
    cached_input: int | None          # subset of logical_input
    cache_creation_input: int | None  # Anthropic only; subset of logical_input
    reasoning_output: int | None      # subset of logical_output; None = not reported
    notes: list[str] = field(default_factory=list)
    doc_source: str = ""
    checked_at: str = CHECKED_AT

    def to_dict(self) -> dict:
        return asdict(self)


def _int(d: dict | None, key: str) -> int | None:
    if not isinstance(d, dict):
        return None
    v = d.get(key)
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def unknown(provider: str, schema: str, reason: str) -> NormalizedUsage:
    return NormalizedUsage(provider=provider, schema=schema, status="unknown", logical_input=None,
                           logical_output=None, logical_total=None, cached_input=None,
                           cache_creation_input=None, reasoning_output=None, notes=[reason],
                           doc_source=DOC_SOURCES.get(schema, ""))


def normalize_openai_responses(usage: dict | None) -> NormalizedUsage:
    if not usage:
        return unknown("openai", OPENAI_SCHEMA, "usage object absent")
    inp, out = _int(usage, "input_tokens"), _int(usage, "output_tokens")
    cached = _int(usage.get("input_tokens_details"), "cached_tokens")
    reasoning = _int(usage.get("output_tokens_details"), "reasoning_tokens")
    notes = []
    if cached is None:
        notes.append("input_tokens_details.cached_tokens absent")
    if reasoning is None:
        notes.append("output_tokens_details.reasoning_tokens absent (not evidence of no reasoning)")
    status = "ok" if inp is not None and out is not None else ("partial" if inp is not None or out is not None else "unknown")
    if status != "ok":
        notes.append("input_tokens/output_tokens incomplete")
    total = inp + out if inp is not None and out is not None else None
    return NormalizedUsage(provider="openai", schema=OPENAI_SCHEMA, status=status, logical_input=inp,
                           logical_output=out, logical_total=total, cached_input=cached,
                           cache_creation_input=None, reasoning_output=reasoning, notes=notes,
                           doc_source=DOC_SOURCES[OPENAI_SCHEMA])


def normalize_anthropic_messages(usage: dict | None) -> NormalizedUsage:
    if not usage:
        return unknown("anthropic", ANTHROPIC_SCHEMA, "usage object absent")
    uncached = _int(usage, "input_tokens")
    created = _int(usage, "cache_creation_input_tokens")
    read = _int(usage, "cache_read_input_tokens")
    out = _int(usage, "output_tokens")
    notes = []
    if uncached is None or out is None:
        return NormalizedUsage(provider="anthropic", schema=ANTHROPIC_SCHEMA, status="unknown",
                               logical_input=None, logical_output=out, logical_total=None,
                               cached_input=read, cache_creation_input=created, reasoning_output=None,
                               notes=["input_tokens or output_tokens absent"], doc_source=DOC_SOURCES[ANTHROPIC_SCHEMA])
    status = "ok"
    if created is None:
        created_v, status = 0, "partial"
        notes.append("cache_creation_input_tokens absent; counted as 0 only for the sum, flagged partial")
    else:
        created_v = created
    if read is None:
        read_v, status = 0, "partial"
        notes.append("cache_read_input_tokens absent; counted as 0 only for the sum, flagged partial")
    else:
        read_v = read
    logical_in = uncached + created_v + read_v
    notes.append("reasoning tokens are not reported separately by this API")
    return NormalizedUsage(provider="anthropic", schema=ANTHROPIC_SCHEMA, status=status,
                           logical_input=logical_in, logical_output=out, logical_total=logical_in + out,
                           cached_input=read, cache_creation_input=created, reasoning_output=None,
                           notes=notes, doc_source=DOC_SOURCES[ANTHROPIC_SCHEMA])


NORMALIZERS = {
    OPENAI_SCHEMA: normalize_openai_responses,
    ANTHROPIC_SCHEMA: normalize_anthropic_messages,
}


def normalize(schema: str, usage: dict | None) -> NormalizedUsage:
    if schema not in NORMALIZERS:
        raise ValueError(f"no usage normalizer for schema {schema!r}")
    return NORMALIZERS[schema](usage)
