"""USD list-price cost accounting (metrics.cost), adaptive per-provider
concurrency (orchestration.scheduler), the public TileLang Reference Skill
gate and credential hygiene of artifacts."""
from __future__ import annotations

import json
import threading
import time

import pytest

from tilebench.llm.v2.devtools import cost_report, synthetic_context
from tilebench.llm.v2.evaluation.launcher import MockEvaluator
from tilebench.llm.v2.metrics import cost as C
from tilebench.llm.v2.metrics import efficiency as e
from tilebench.llm.v2.orchestration.campaign import ResumeRefused, _check_resume, new_trajectory_state, protocol_identity
from tilebench.llm.v2.orchestration.runner import RunnerConfig, TrajectoryRunner
from tilebench.llm.v2.orchestration.scheduler import AdaptiveLimiter, GatedProvider, classify_failure, header_headroom
from tilebench.llm.v2.manifests import schema as ms
from tilebench.llm.v2.prompts.renderer import render_initial, render_refinement, render_system
from tilebench.llm.v2.providers.base import GenerationResult, TransportError, retry_after_seconds
from tilebench.llm.v2.providers.factory import generator_spec
from tilebench.llm.v2.providers.mock import scripted_text
from tilebench.llm.v2.providers.usage import normalize_openai_responses
from tilebench.llm.v2.skills import loader
from tilebench.llm.v2.tasks.support import eligibility

OPENAI_USAGE = {"input_tokens": 100000, "input_tokens_details": {"cached_tokens": 20000, "cache_write_tokens": 30000},
                "output_tokens": 50000, "output_tokens_details": {"reasoning_tokens": 40000}, "total_tokens": 150000}
ANTHROPIC_USAGE = {"input_tokens": 10000, "cache_creation_input_tokens": 3000,
                   "cache_creation": {"ephemeral_5m_input_tokens": 2000, "ephemeral_1h_input_tokens": 1000},
                   "cache_read_input_tokens": 5000, "output_tokens": 60000, "service_tier": "standard", "inference_geo": "global"}


# ----------------------------------------------------------------------------- cost formulas
def test_openai_cost_uncached_cached_write_output_and_reasoning_not_double_counted():
    r = C.request_cost("openai", "gpt-6.1-sol", OPENAI_USAGE, service_tier="default")
    assert r["usd_cost_status"] == "exact_from_usage_and_frozen_price" and r["pricing_rule_id"].endswith("short-context/2026-10-06.1")
    assert r["tokens"]["uncached_input"] == 50000 and r["tokens"]["cached_input"] == 20000 and r["tokens"]["cache_write"] == 30000
    assert r["usd_uncached_input"] == pytest.approx(0.10) and r["usd_cached_input"] == pytest.approx(0.002)
    assert r["usd_cache_write"] == pytest.approx(0.075) and r["usd_output"] == pytest.approx(0.50)
    assert r["estimated_cost_usd"] == pytest.approx(0.677)                          # reasoning (40k) is inside output (50k)
    long = C.request_cost("openai", "gpt-6.1-sol", {**OPENAI_USAGE, "input_tokens": 400000}, service_tier="default")
    assert long["pricing_rule_id"].endswith("long-context/2026-10-06.1") and long["usd_uncached_input"] == pytest.approx(350000 * 4 / 1e6)
    flex = C.request_cost("openai", "gpt-6.1-sol", OPENAI_USAGE, service_tier="flex")
    assert flex["usd_cost_status"] == "unknown" and flex["estimated_cost_usd"] is None          # never a guessed tier


def test_anthropic_cost_uncached_cache_write_5m_1h_read_output_thinking_not_double_counted():
    r = C.request_cost("anthropic", "claude-opus-5-5", ANTHROPIC_USAGE)
    assert r["usd_cost_status"] == "exact_from_usage_and_frozen_price"
    assert r["usd_uncached_input"] == pytest.approx(0.04) and r["usd_cache_creation_5m"] == pytest.approx(0.01)
    assert r["usd_cache_creation_1h"] == pytest.approx(0.008) and r["usd_cache_read"] == pytest.approx(0.001)
    assert r["usd_output"] == pytest.approx(1.2) and r["estimated_cost_usd"] == pytest.approx(1.259)
    us = C.request_cost("anthropic", "claude-opus-5-5", {**ANTHROPIC_USAGE, "inference_geo": "us"})
    assert us["usd_cost_status"] == "unknown"                                       # 1.1x regime is not in the snapshot


def test_unknown_usage_is_never_zero_and_partial_usage_is_a_lower_bound():
    unknown = C.transport_cost({"outcome": "failed", "charged": "unknown"}, "openai", "gpt-6.1-sol")
    assert unknown["usd_cost_status"] == "unknown" and unknown["estimated_cost_usd"] is None
    partial = C.transport_cost({"outcome": "failed", "charged": "unknown",
                                "usage_partial": {"input_tokens": 23629, "output_tokens": 5, "service_tier": "standard",
                                                  "inference_geo": "global", "cache_creation_input_tokens": 0,
                                                  "cache_read_input_tokens": 0}}, "anthropic", "claude-opus-5-5")
    assert partial["usd_cost_status"] == "partial_lower_bound" and partial["estimated_cost_usd"] is None
    assert partial["estimated_cost_lower_bound_usd"] == pytest.approx(23629 * 4 / 1e6 + 5 * 20 / 1e6)
    rejected = C.transport_cost({"outcome": "failed", "charged": "no"}, "openai", "gpt-6.1-sol")
    assert rejected["usd_cost_status"] == "not_sent" and rejected["estimated_cost_usd"] == 0.0


def test_retry_costs_accumulate_and_unknown_makes_the_total_inexact():
    a = {"usd_cost_status": "exact_from_usage_and_frozen_price", "estimated_cost_usd": 0.5}
    b = {"usd_cost_status": "not_sent", "estimated_cost_usd": 0.0}
    c = {"usd_cost_status": "exact_from_usage_and_frozen_price", "estimated_cost_usd": 0.2}
    assert C.combine([a, b, c])["estimated_cost_usd"] == pytest.approx(0.7)
    u = {"usd_cost_status": "unknown", "estimated_cost_usd": None, "estimated_cost_lower_bound_usd": None}
    tot = C.combine([a, u])
    assert tot["estimated_cost_usd"] is None and tot["estimated_cost_lower_bound_usd"] == pytest.approx(0.5)
    assert tot["usd_cost_status"] == "unknown" and tot["unknown_records"] == 1


def test_pricing_manifest_is_validated_and_hash_pinned():
    p = C.load_pricing()
    assert p["currency"] == "USD" and all(s["url"].startswith("https://") for s in p["providers"]["openai"]["sources"])
    b = C.pricing_binding()
    assert b["pricing_snapshot_sha256"] == C.pricing_sha256() and len(b["pricing_snapshot_sha256"]) == 64
    bad = json.loads(json.dumps(p))
    bad["providers"]["openai"]["models"]["gpt-6.1-sol"]["rules"][0]["rates"].pop("cache_write")
    with pytest.raises(C.PricingError):
        C.validate_pricing(bad)


# ----------------------------------------------------------------------------- runner integration
class FakeOpenAI:
    name = "openai"
    usage_schema = "openai_responses_v1"

    def __init__(self, items):
        self.items, self.requests = list(items), []

    def generate(self, req):
        self.requests.append(req)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return GenerationResult(text=item, model_id="gpt-6.1-sol", provider="openai", response_id=f"r{len(self.requests)}",
                                usage_raw=OPENAI_USAGE, usage=normalize_openai_responses(OPENAI_USAGE), transport_attempts=1,
                                elapsed_s=0.0, raw_response={"service_tier": "default"}, terminal_status="completed",
                                rate_limit={"x-ratelimit-limit-tokens": "1000000", "x-ratelimit-remaining-tokens": "900000"})


def _runner(study, folds, tmp_path, provider, run_type="validation"):
    ctx, job, _ = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    el = eligibility("vector_add", "fp16", "B200", "triton", study, folds)
    st = new_trajectory_state(el, ctx, "gpt", "base", "h", "c", "t", run_type=run_type, campaign="t",
                              protocol=protocol_identity(study), pricing_binding=C.pricing_binding())
    cfg = RunnerConfig(model_id="gpt-6.1-sol", provider_name="openai", settings={}, feedback_limits=study["feedback"],
                       retry_backoff_s=0.0)
    return TrajectoryRunner(state=st, tdir=tmp_path / "t", ctx=ctx, provider=provider, evaluator=MockEvaluator(), job=job,
                            cfg=cfg, sleep=lambda s: None), st


def test_failed_candidate_keeps_its_dollar_cost_and_retries_add_up(study, folds, tmp_path):
    good = scripted_text("impl_triton.py", "# MOCK: valid 1.0\ndef run(*a): pass\ndef get_last_config(): return {}")
    prov = FakeOpenAI(["no fenced file here",                                         # round 1: format error
                       TransportError("boom", charged="no", status_code=503), good])   # round 2: rejected retry + success
    r, st = _runner(study, folds, tmp_path, prov)
    r.step()
    a1 = st.rounds[0].attempts[0]
    assert st.rounds[0].status == "format_error" and a1.usd_cost == pytest.approx(0.677)
    assert a1.usd_cost_status == "exact_from_usage_and_frozen_price"
    r.step()
    a2 = st.rounds[1].attempts[0]
    assert a2.transport_attempts == 2 and a2.usd_cost == pytest.approx(0.677)        # the 503 rejection is not billed
    r.save()
    cs = st.cost_summary
    assert cs["known_estimated_usd"] == pytest.approx(1.354) and cs["usd_cost_exact"] and cs["unknown_cost_attempts"] == 0
    assert cs["rounds"][1]["cumulative_estimated_usd"] == pytest.approx(1.354)
    assert cs["rounds"][1]["cumulative_logical_tokens"] == 300000
    ledger = [json.loads(x) for x in (tmp_path / "t" / "usage.jsonl").read_text().splitlines()]
    assert all(row["pricing_snapshot_sha256"] == C.pricing_sha256() and row["usd_cost_status"] for row in ledger)
    rows = [json.loads(x) for x in (tmp_path / "t" / "transport.jsonl").read_text().splitlines()]
    assert all("usd" in x for x in rows if x["event"] != "sending")
    assert any(x.get("rate_limit") for x in rows if x["event"] == "succeeded")


def test_unknown_charge_transport_failure_makes_attempt_usd_unknown(study, folds, tmp_path):
    good = scripted_text("impl_triton.py", "# MOCK: valid 1.0\ndef run(*a): pass\ndef get_last_config(): return {}")
    prov = FakeOpenAI([TransportError("stream cut", charged="unknown", stream_events=5), good])
    r, st = _runner(study, folds, tmp_path, prov)
    r.step()
    a = st.rounds[0].attempts[0]
    assert a.usd_cost is None and a.usd_cost_status == "unknown" and a.usd_cost_lower_bound == pytest.approx(0.677)
    r.save()
    assert st.cost_summary["unknown_cost_attempts"] == 1 and st.cost_summary["rounds"][0]["cumulative_estimated_usd"] is None


def test_formal_resume_refuses_a_changed_pricing_snapshot(study, folds, tmp_path):
    r, st = _runner(study, folds, tmp_path, FakeOpenAI([]), run_type="formal")
    gen = generator_spec(ms.load_models(), "gpt")
    st.generator = gen.record()
    other = {**C.pricing_binding(), "pricing_snapshot_sha256": "0" * 64}
    with pytest.raises(ResumeRefused, match="pricing"):
        _check_resume(st, config_hash="h", content_hashes=st.content_hashes, run_type="formal", gen=gen, pricing_binding=other)
    _check_resume(st, config_hash="h", content_hashes=st.content_hashes, run_type="formal", gen=gen, pricing_binding=C.pricing_binding())


def test_pricing_and_rate_limits_never_enter_a_prompt(study, folds):
    ctx, _, _ = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    texts = [render_system(ctx), render_initial(ctx),
             render_refinement(ctx, round_index=2, prev={"round": 1, "status": "numerical_error", "source": "x", "config": {},
                                                         "latency_ms_mean": None, "latency_ms_samples": None,
                                                         "diagnostic": "usd 0.677 pricing x-ratelimit-remaining"},
                               best_valid=None, history=[], limits=study["feedback"])]
    for t in texts:
        for word in ("pricing_snapshot", "estimated_cost", "x-ratelimit", "anthropic-ratelimit", "api_pricing"):
            assert word not in t
        for rate in ("$2.00", "$10.00", "$20 / MTok", "0.677"):
            assert rate not in t.split("# Optimization round")[-1] or rate not in t


def test_e_usd_uses_a_dollar_budget_independently_of_tokens():
    rounds = [{"round": 1, "attempts": [{"cost": 1000, "usd": 0.5}], "valid": True, "latency_ms": 2.0},
              {"round": 2, "attempts": [{"cost": 1000, "usd": 3.0}], "valid": True, "latency_ms": 1.0}]
    tok = e.curve(1.0, rounds)
    usd = e.curve(1.0, rounds, cost_key="usd")
    assert e.efficiency_at(tok, 2000) == 1.0 and e.efficiency_at(usd, 1.0) == 0.5 and e.efficiency_at(usd, 3.5) == 1.0
    unk = e.curve(1.0, [{"round": 1, "attempts": [{"cost": 10, "usd": None}], "valid": True, "latency_ms": 1.0}], cost_key="usd")
    assert e.efficiency_at(unk, 100.0) is None and e.efficiency_at(e.curve(1.0, unk and [{"round": 1, "attempts": [{"cost": 10}],
                                                                                         "valid": True, "latency_ms": 1.0}]), 100) == 1.0


def test_cost_report_aggregates_by_provider_model_dsl_operator(study, folds, tmp_path):
    good = scripted_text("impl_triton.py", "# MOCK: valid 1.0\ndef run(*a): pass\ndef get_last_config(): return {}")
    r, st = _runner(study, folds, tmp_path, FakeOpenAI([good]))
    st.generator = {"provider": "openai", "model_id": "gpt-6.1-sol"}
    r.step(); r.step(); r.save()
    rep = cost_report(tmp_path)
    g = rep["groups"]
    assert g["campaign"]["known_estimated_usd"] == pytest.approx(0.677) and g["provider=openai"]["trajectories"] == 1
    assert g["campaign"]["provider"] == "openai"       # "mixed" only when a group spans providers
    assert "model=gpt-6.1-sol/dsl=triton/operator=vector_add" in g and rep["pricing_snapshot_sha256"] == [C.pricing_sha256()]


def test_no_credential_ever_reaches_artifacts(study, folds, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-SECRET-should-never-appear")
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-admin-SECRET-should-never-appear")
    good = scripted_text("impl_triton.py", "# MOCK: valid 1.0\ndef run(*a): pass\ndef get_last_config(): return {}")
    r, st = _runner(study, folds, tmp_path, FakeOpenAI([good]))
    r.run(max_steps=2)
    from tilebench.llm.v2.providers import billing

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"results": [{"amount": {"value": 1.25, "currency": "usd"}}]}]}
    seen = {}
    billing.reconcile(tmp_path / "t", start=0, end=1, providers=("openai",),
                      http_get=lambda url, **kw: seen.setdefault("h", kw["headers"]) and Resp())
    blob = "".join(p.read_text(errors="replace") for p in (tmp_path / "t").rglob("*") if p.is_file())
    assert "SECRET" not in blob and "sk-" not in blob
    status = json.loads((tmp_path / "t" / "billing_reconciliation" / "status.json").read_text())
    assert status["providers"]["openai"]["provider_billed_usd"] == 1.25 and status["providers"]["openai"]["credential_env"] == "OPENAI_ADMIN_KEY"
    monkeypatch.delenv("OPENAI_ADMIN_KEY")
    st2 = billing.reconcile(tmp_path / "u", start=0, end=1, providers=("openai",))
    assert st2["providers"]["openai"]["status"] == "unavailable"


# ----------------------------------------------------------------------------- scheduler
def test_header_headroom_and_retry_after():
    oa = {"x-ratelimit-limit-tokens": "1000", "x-ratelimit-remaining-tokens": "400", "x-ratelimit-limit-requests": "100",
          "x-ratelimit-remaining-requests": "90", "x-ratelimit-reset-tokens": "6m0s"}
    assert header_headroom(oa) == pytest.approx(0.4)
    an = {"anthropic-ratelimit-output-tokens-limit": "400000", "anthropic-ratelimit-output-tokens-remaining": "300000"}
    assert header_headroom(an) == pytest.approx(0.75) and header_headroom({"date": "x"}) is None
    assert retry_after_seconds({"retry-after": "7"}) == 7.0
    assert retry_after_seconds({**oa, "x-ratelimit-remaining-tokens": "0"}) == 360.0
    assert retry_after_seconds(oa) == 0.0                                   # nothing exhausted, no Retry-After


def _limiter(**kw):
    clock = {"t": 1000.0}
    lim = AdaptiveLimiter("openai", initial=3, max_limit=24, ramp_min_interval_s=60, clock=lambda: clock["t"],
                          host_ok=lambda: (True, "test"), log=lambda s: None, **kw)
    return lim, clock


def _ok(lim, n, headers):
    for _ in range(n):
        lim.acquire()
        lim.release("ok", headers=headers)


def test_limiter_ramps_only_with_real_headers_and_headroom():
    lim, clock = _limiter()
    _ok(lim, 5, None)
    clock["t"] += 120
    _ok(lim, 3, None)
    assert lim.limit == 3                                                   # no headers seen: no ramp
    good = {"x-ratelimit-limit-tokens": "1000", "x-ratelimit-remaining-tokens": "900"}
    _ok(lim, 3, good)
    assert lim.limit == 6                                                   # 3 -> 6
    clock["t"] += 120
    _ok(lim, 6, good)
    assert lim.limit == 12                                                  # 6 -> 12
    clock["t"] += 120
    tight = {"x-ratelimit-limit-tokens": "1000", "x-ratelimit-remaining-tokens": "300"}
    _ok(lim, 12, tight)
    assert lim.limit == 12                                                  # headroom < 0.5: hold


def test_limiter_429_halves_and_pauses_and_sustained_overload_backs_off():
    lim, clock = _limiter()
    lim.limit = 12
    lim.acquire()
    lim.release("rate_limited", headers={"retry-after": "30"})
    assert lim.limit == 6 and lim.pause_until == pytest.approx(clock["t"] + 30)
    lim2, clock2 = _limiter()
    lim2.limit = 12
    for i in range(2):
        lim2.acquire(); lim2.release("overloaded")
    assert lim2.limit == 12                                                 # isolated overloads: recorded only
    lim2.acquire(); lim2.release("overloaded")
    assert lim2.limit == 6                                                  # sustained (3 of 3 in 10 min): halve
    for i in range(3):
        lim2.acquire(); lim2.release("overloaded")
    assert lim2.limit == 6                                                  # at most one overload backoff per window
    assert classify_failure(TransportError("x", charged="no", status_code=429)) == "rate_limited"
    assert classify_failure(TransportError("APIError: Our servers are currently overloaded.", charged="unknown")) == "overloaded"


def test_gated_provider_caps_in_flight_requests_and_the_global_eval_budget_is_a_hard_bound():
    from tilebench.llm.v2.orchestration.scheduler import EvalBudget
    lim = AdaptiveLimiter("openai", initial=3, max_limit=3, max_pending=2, host_ok=lambda: (True, ""), log=lambda s: None)
    state = {"cur": 0, "max": 0}
    lock = threading.Lock()

    class Slow:
        name, usage_schema = "openai", "openai_responses_v1"

        def generate(self, req):
            with lock:
                state["cur"] += 1
                state["max"] = max(state["max"], state["cur"])
            time.sleep(0.05)
            with lock:
                state["cur"] -= 1
            return GenerationResult(text="x", model_id="m", provider="openai", response_id="r", usage_raw={},
                                    usage=normalize_openai_responses({"input_tokens": 1, "output_tokens": 1}),
                                    transport_attempts=1, elapsed_s=0.0)
    gp = GatedProvider(Slow(), lim)
    threads = [threading.Thread(target=gp.generate, args=(None,)) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert state["max"] == 3                                               # never more than the current limit in flight
    # hard global budget: a slot is reserved before sending and held until the round closes (release by the runner)
    budget = EvalBudget(2)
    gb = GatedProvider(Slow(), AdaptiveLimiter("anthropic", initial=8, max_limit=8, host_ok=lambda: (True, ""),
                                               log=lambda s: None), budget=budget)
    held, release = [], threading.Event()

    def trajectory():
        gb.before_send(); gb.generate(None)                               # generation done; candidate "waits for evaluation"
        held.append(1)
        release.wait(5)
        budget.release()                                                   # the round closed
    ts = [threading.Thread(target=trajectory) for _ in range(5)]
    [t.start() for t in ts]
    time.sleep(0.5)
    assert budget.used == 2 and len(held) == 2 and budget.max_used == 2    # 3 trajectories wait; never more than the cap
    release.set()
    [t.join(timeout=10) for t in ts]
    assert len(held) == 5 and budget.used == 0 and budget.max_used == 2 and budget.reservations == 5


# ----------------------------------------------------------------------------- TileLang public reference gate
def test_tilelang_public_reference_is_licensed_sendable_and_the_internal_one_is_not():
    m = loader.load_manifest()
    e_pub = m["reference"]["tilelang"]["0.1.11"]
    assert e_pub["permission"] == "public" and e_pub["status"] == "approved"
    assert sorted(e_pub["sendable_to"]) == ["anthropic", "openai"] and e_pub["publishable"] is True
    prov = json.loads((loader.REPO_ROOT / e_pub["asset_manifest"]).read_text())
    assert prov["license"]["name"] == "MIT" and prov["upstream"]["commit"] == "cd37ed5fc35ae7a60a1277c8eb49028174ac51e6"
    assert any("tilelang-guide" in x for x in prov["forbidden_sources_not_used"])
    for provider in ("openai", "anthropic"):
        comp = loader.load_component(m, "reference", "tilelang", "0.1.11", provider=provider)
        assert "MIT License" in comp.text and "Copyright (c) Tile-AI." in comp.text
    internal = m["reference"]["tilelang"]["0.1.11-internal-superseded"]
    assert internal["sendable_to"] == [] and internal["permission"] == "internal"
    with pytest.raises(loader.SkillPermissionError):
        loader.load_component(m, "reference", "tilelang", "0.1.11-internal-superseded", provider="openai", require_status=())
    assert ms.load_study()["dsls"]["tilelang"]["reference_version"] == "0.1.11"


def test_stop_file_pauses_before_anything_is_sent_and_resume_has_no_orphan(study, folds, tmp_path):
    good = scripted_text("impl_triton.py", "# MOCK: valid 1.0\ndef run(*a): pass\ndef get_last_config(): return {}")
    stop = {"on": False}
    lim = AdaptiveLimiter("openai", initial=3, max_limit=3, host_ok=lambda: (True, ""), log=lambda s: None,
                          stop_fn=lambda: stop["on"])
    inner = FakeOpenAI([good, good])
    r, st = _runner(study, folds, tmp_path, GatedProvider(inner, lim))
    r.step(); r.step()                                                  # round 1 generated and evaluated
    stop["on"] = True
    assert r.run() == "paused" and len(inner.requests) == 1
    rows = [json.loads(x) for x in (tmp_path / "t" / "transport.jsonl").read_text().splitlines()]
    assert [x["event"] for x in rows] == ["sending", "succeeded"]       # no dangling `sending` for round 2
    assert lim.in_flight == 0
    stop["on"] = False
    st2 = type(st).load(tmp_path / "t" / "trajectory.json")
    r2 = TrajectoryRunner(state=st2, tdir=tmp_path / "t", ctx=r.ctx, provider=GatedProvider(inner, lim), evaluator=MockEvaluator(),
                          job=r.job, cfg=r.cfg, sleep=lambda s: None)
    r2.step(); r2.step()
    assert st2.rounds[1].status == "valid" and len(inner.requests) == 2
    rows = [json.loads(x) for x in (tmp_path / "t" / "transport.jsonl").read_text().splitlines()]
    assert not any(x["event"] == "orphaned" for x in rows)
