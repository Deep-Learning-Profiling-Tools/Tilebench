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
interpreted as "the model did not reason".

Normalized record schema `normalized_usage/2` (symmetric, nullable; the raw
provider usage is archived beside it, untouched):
    provider, model, response_id, service_tier, inference_geo,
    logical_input_total, uncached_input, cached_input, cache_write_input,
    logical_output_total, reasoning_or_thinking_output (SUBSET of output),
    non_reasoning_output (= output - reasoning/thinking when both are reported),
    total_tokens_if_reported, server_tool_counts, cache_diagnostics.
OpenAI: input_tokens_details.cache_write_tokens and
output_tokens_details.reasoning_tokens are read when returned (both observed
in the 2026-10-06 responses); prompt-cache diagnostics (prompt_cache_key,
prompt_cache_retention, any other input_tokens_details field) are copied when
present. Anthropic: output_tokens_details.thinking_tokens, server_tool_use,
service_tier, inference_geo and a non-null `diagnostics` are read when the
API returns them (thinking was NOT reported separately in the 2026-10-06
responses; it is then None, never inferred). Reasoning / thinking tokens are
part of the output count and are never added again."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

OPENAI_SCHEMA = "openai_responses_v1"
ANTHROPIC_SCHEMA = "anthropic_messages_v1"
CHECKED_AT = "2026-10-05"
DOC_SOURCES = {
    OPENAI_SCHEMA: "https://platform.openai.com/docs/api-reference/responses/object#usage",
    ANTHROPIC_SCHEMA: "https://docs.anthropic.com/en/api/messages#response-usage",
}


NORMALIZED_SCHEMA = "normalized_usage/2"


@dataclass
class NormalizedUsage:
    provider: str
    schema: str
    status: str                       # ok | partial | unknown
    logical_input: int | None
    logical_output: int | None
    logical_total: int | None
    cached_input: int | None          # subset of logical_input (cache reads)
    cache_creation_input: int | None  # subset of logical_input (cache writes)
    reasoning_output: int | None      # subset of logical_output; None = not reported
    notes: list[str] = field(default_factory=list)
    doc_source: str = ""
    checked_at: str = CHECKED_AT
    # normalized_usage/2 (symmetric, nullable)
    schema_version: str = NORMALIZED_SCHEMA
    model: str | None = None
    response_id: str | None = None
    service_tier: str | None = None
    inference_geo: str | None = None
    logical_input_total: int | None = None
    uncached_input: int | None = None
    cache_write_input: int | None = None
    logical_output_total: int | None = None
    reasoning_or_thinking_output: int | None = None
    non_reasoning_output: int | None = None
    total_tokens_if_reported: int | None = None
    server_tool_counts: dict | None = None
    cache_diagnostics: dict | None = None

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


def _response_meta(response: dict | None) -> dict:
    r = response if isinstance(response, dict) else {}
    return {"model": r.get("model") if isinstance(r.get("model"), str) else None,
            "response_id": r.get("id") if isinstance(r.get("id"), str) else None}


def normalize_openai_responses(usage: dict | None, response: dict | None = None) -> NormalizedUsage:
    if not usage:
        return unknown("openai", OPENAI_SCHEMA, "usage object absent")
    in_det = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
    out_det = usage.get("output_tokens_details") if isinstance(usage.get("output_tokens_details"), dict) else {}
    inp, out = _int(usage, "input_tokens"), _int(usage, "output_tokens")
    cached = _int(in_det, "cached_tokens")
    write = _int(in_det, "cache_write_tokens")
    reasoning = _int(out_det, "reasoning_tokens")
    total_rep = _int(usage, "total_tokens")
    notes = []
    if cached is None:
        notes.append("input_tokens_details.cached_tokens absent")
    if write is None:
        notes.append("input_tokens_details.cache_write_tokens absent")
    if reasoning is None:
        notes.append("output_tokens_details.reasoning_tokens absent (not evidence of no reasoning)")
    status = "ok" if inp is not None and out is not None else ("partial" if inp is not None or out is not None else "unknown")
    if status != "ok":
        notes.append("input_tokens/output_tokens incomplete")
    total = inp + out if inp is not None and out is not None else None
    if total is not None and total_rep is not None and total_rep != total:
        notes.append(f"total_tokens {total_rep} != input_tokens + output_tokens {total} (the sum is used)")
    r = response if isinstance(response, dict) else {}
    diag = {k: r[k] for k in ("prompt_cache_key", "prompt_cache_retention") if k in r}
    extra = {k: v for k, v in in_det.items() if k not in ("cached_tokens", "cache_write_tokens")}
    if extra:
        diag["input_tokens_details_other"] = extra
    uncached = inp - (cached or 0) - (write or 0) if inp is not None and cached is not None and write is not None else None
    return NormalizedUsage(provider="openai", schema=OPENAI_SCHEMA, status=status, logical_input=inp,
                           logical_output=out, logical_total=total, cached_input=cached,
                           cache_creation_input=write, reasoning_output=reasoning, notes=notes,
                           doc_source=DOC_SOURCES[OPENAI_SCHEMA], **_response_meta(response),
                           service_tier=r.get("service_tier") if isinstance(r.get("service_tier"), str) else None,
                           logical_input_total=inp, uncached_input=uncached, cache_write_input=write,
                           logical_output_total=out, reasoning_or_thinking_output=reasoning,
                           non_reasoning_output=(out - reasoning) if out is not None and reasoning is not None else None,
                           total_tokens_if_reported=total_rep, server_tool_counts=None, cache_diagnostics=diag or None)


def normalize_anthropic_messages(usage: dict | None, response: dict | None = None) -> NormalizedUsage:
    if not usage:
        return unknown("anthropic", ANTHROPIC_SCHEMA, "usage object absent")
    uncached = _int(usage, "input_tokens")
    created = _int(usage, "cache_creation_input_tokens")
    read = _int(usage, "cache_read_input_tokens")
    out = _int(usage, "output_tokens")
    out_det = usage.get("output_tokens_details") if isinstance(usage.get("output_tokens_details"), dict) else {}
    thinking = _int(out_det, "thinking_tokens")
    r = response if isinstance(response, dict) else {}
    stu = usage.get("server_tool_use") if isinstance(usage.get("server_tool_use"), dict) else None
    diag = {}
    if r.get("diagnostics") is not None:
        diag["diagnostics"] = r.get("diagnostics")
    if isinstance(usage.get("cache_creation"), dict):
        diag["cache_creation"] = usage["cache_creation"]
    for k, v in usage.items():
        if "cache" in k and k not in ("cache_creation", "cache_creation_input_tokens", "cache_read_input_tokens"):
            diag[k] = v
    meta = dict(**_response_meta(response),
                service_tier=usage.get("service_tier") if isinstance(usage.get("service_tier"), str) else None,
                inference_geo=usage.get("inference_geo") if isinstance(usage.get("inference_geo"), str) else None,
                uncached_input=uncached, cache_write_input=created, logical_output_total=out,
                reasoning_or_thinking_output=thinking,
                non_reasoning_output=(out - thinking) if out is not None and thinking is not None else None,
                total_tokens_if_reported=None, server_tool_counts=stu, cache_diagnostics=diag or None)
    notes = []
    if uncached is None or out is None:
        return NormalizedUsage(provider="anthropic", schema=ANTHROPIC_SCHEMA, status="unknown",
                               logical_input=None, logical_output=out, logical_total=None,
                               cached_input=read, cache_creation_input=created, reasoning_output=thinking,
                               notes=["input_tokens or output_tokens absent"], doc_source=DOC_SOURCES[ANTHROPIC_SCHEMA],
                               logical_input_total=None, **meta)
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
    if thinking is None:
        notes.append("thinking tokens not reported separately in this response (included in output_tokens)")
    return NormalizedUsage(provider="anthropic", schema=ANTHROPIC_SCHEMA, status=status,
                           logical_input=logical_in, logical_output=out, logical_total=logical_in + out,
                           cached_input=read, cache_creation_input=created, reasoning_output=thinking,
                           notes=notes, doc_source=DOC_SOURCES[ANTHROPIC_SCHEMA], logical_input_total=logical_in, **meta)


NORMALIZERS = {
    OPENAI_SCHEMA: normalize_openai_responses,
    ANTHROPIC_SCHEMA: normalize_anthropic_messages,
}


def normalize(schema: str, usage: dict | None, response: dict | None = None) -> NormalizedUsage:
    if schema not in NORMALIZERS:
        raise ValueError(f"no usage normalizer for schema {schema!r}")
    return NORMALIZERS[schema](usage, response)
