"""Usage normalization and the mock provider."""
import pytest

from tilebench.llm.v2.providers import usage as u
from tilebench.llm.v2.providers.base import GenerationRequest, TransportError
from tilebench.llm.v2.providers.ledger import ledger_totals
from tilebench.llm.v2.providers.mock import MockProvider


def test_openai_cached_and_reasoning_are_subsets_not_added():
    n = u.normalize_openai_responses({"input_tokens": 1000, "output_tokens": 300, "total_tokens": 1300,
                                      "input_tokens_details": {"cached_tokens": 800},
                                      "output_tokens_details": {"reasoning_tokens": 250}})
    assert (n.logical_input, n.logical_output, n.logical_total) == (1000, 300, 1300)
    assert n.cached_input == 800 and n.reasoning_output == 250 and n.status == "ok"


def test_anthropic_logical_input_includes_cache_fields():
    n = u.normalize_anthropic_messages({"input_tokens": 100, "cache_creation_input_tokens": 2000,
                                        "cache_read_input_tokens": 5000, "output_tokens": 400,
                                        "cache_creation": {"ephemeral_5m_input_tokens": 2000, "ephemeral_1h_input_tokens": 0}})
    assert n.logical_input == 7100 and n.logical_total == 7500 and n.cached_input == 5000
    assert n.reasoning_output is None and n.status == "ok"


def test_missing_usage_is_unknown_not_zero():
    n = u.normalize_openai_responses(None)
    assert n.status == "unknown" and n.logical_total is None
    p = u.normalize_openai_responses({"input_tokens": 10})
    assert p.status == "partial" and p.logical_total is None
    a = u.normalize_anthropic_messages({"input_tokens": 10, "output_tokens": 5})
    assert a.status == "partial" and a.logical_input == 10 and "cache_creation_input_tokens absent" in " ".join(a.notes)


def test_missing_reasoning_count_is_not_no_reasoning():
    n = u.normalize_openai_responses({"input_tokens": 1, "output_tokens": 1})
    assert n.reasoning_output is None and any("not evidence of no reasoning" in x for x in n.notes)


def test_ledger_totals_flag_unknown_rows():
    rows = [{"usage": {"status": "ok", "logical_input": 10, "logical_output": 5, "logical_total": 15}},
            {"usage": {"status": "unknown", "logical_total": None}}]
    t = ledger_totals(rows)
    assert t["logical_total"] == 15 and t["rows_with_unknown_usage"] == 1 and t["cost_exact"] is False


def test_mock_provider_transport_failures_and_usage_withheld():
    p = MockProvider([{"text": "a", "transport_failures": 2}, {"text": "b", "usage_missing": True}])
    req = GenerationRequest(system="s", user="u", model_id="m", provider="mock")
    with pytest.raises(TransportError):
        p.generate(req)
    with pytest.raises(TransportError):
        p.generate(req)
    r = p.generate(req)
    assert r.text == "a" and r.usage.status == "ok"
    r2 = p.generate(req)
    assert r2.text == "b" and r2.usage.status == "unknown" and r2.usage_raw is None
