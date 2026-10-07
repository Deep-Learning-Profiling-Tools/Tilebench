"""Regression tests for the 89140bb3 review findings (R1-R12): scope-aware
static checks, evaluator job schema, worker interface/autotuner/config
checks, provider classification and factory, ledger/cost accounting,
metrics unknown-vs-zero, distillation binding, grants, timing normalization,
oracle aliasing, violation-round fallback, isolation report, publication."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tilebench.llm.v2.devtools import synthetic_context
from tilebench.llm.v2.evaluation import anticache
from tilebench.llm.v2.evaluation.job import EvaluationJob, build_evaluation_job, timing_settings, validate_worker_job
from tilebench.llm.v2.evaluation.launcher import MockEvaluator, _bwrap_base, detect_isolation
from tilebench.llm.v2.manifests import schema as ms
from tilebench.llm.v2.metrics import efficiency as e
from tilebench.llm.v2.orchestration import state_machine as sm
from tilebench.llm.v2.orchestration.campaign import new_trajectory_state
from tilebench.llm.v2.orchestration.runner import RunnerConfig, TrajectoryRunner, attempt_cost
from tilebench.llm.v2.orchestration.state import AttemptRecord, TrajectoryState
from tilebench.llm.v2.providers.base import GenerationRequest, GenerationResult, ProviderConfigError, TransportError
from tilebench.llm.v2.providers.ledger import read_jsonl
from tilebench.llm.v2.providers.mock import MockProvider, scripted_text
from tilebench.llm.v2.providers.usage import normalize_openai_responses
from tilebench.llm.v2.tasks.support import eligibility
from tilebench.llm.v2.validation.contract_checks import check_compliance
from tilebench.llm.v2.validation.static_checks import analyze, code_only

KERNEL_ADD = '''import torch
import triton
import triton.language as tl

@triton.jit
def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    out = x + y
    tl.store(out_ptr + offs, out, mask=mask)

def run(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = ((n + 1024 - 1) // 1024,)   # torch.matmul( mentioned in a comment only
    _add[grid](x, y, out, n, BLOCK=1024)
    return out

def get_last_config():
    return {"BLOCK": 1024}
'''

VA_RULES = json.loads(Path("tilebench/llm/v2/contracts/data/vector_add/evaluator_rules.json").read_text())


# ---------------------------------------------------------------- R3: static checks
def test_kernel_arithmetic_is_not_host_delegation():
    res = check_compliance(KERNEL_ADD, "triton", VA_RULES)
    assert res.verdict == "clear", res.to_dict()
    host = KERNEL_ADD.replace("    _add[grid](x, y, out, n, BLOCK=1024)\n    return out", "    out = x + y\n    return out")
    bad = check_compliance(host, "triton", VA_RULES)
    assert bad.verdict == "confirmed_violation" and any("PyTorch + operator" in d for d in bad.diagnostics())


def test_import_alias_autotune_is_confirmed():
    src = "from triton import autotune as tune\nimport triton\n@tune(configs=[], key=['N'])\n@triton.jit\ndef k(): pass\n"
    assert analyze(src, "triton").verdict() == "confirmed_violation"
    src2 = "import triton as tr\n@tr.autotune(configs=[], key=[])\n@tr.jit\ndef k(): pass\n"
    assert analyze(src2, "triton").verdict() == "confirmed_violation"
    src3 = "import torch.nn.functional as F\ndef run(x):\n    return F.softmax(x, dim=-1)\n"
    assert any(ev.category == "delegation" for ev in analyze(src3, "triton").confirmed)


def test_regex_rules_ignore_comments_strings_and_kernel_scope():
    assert "matmul" not in code_only("x = 1  # torch.matmul(\ns = 'torch.matmul('\n")
    rules = {"forbidden_substitutions": [{"pattern": r"torch\.matmul\(", "message": "m", "level": "confirmed", "scope": "host"}],
             "required_evidence": [], "allowed_torch_calls": []}
    assert check_compliance("x = 1  # torch.matmul(\n", "triton", rules).verdict == "clear"
    assert check_compliance('"""docs: torch.matmul( here"""\nx = 1\n', "triton", rules).verdict == "clear"
    # a regex hit on a non-computational line is audit-only (checker v2), never confirmed
    rules2 = {"forbidden_substitutions": [{"pattern": r"sorted", "message": "m", "level": "confirmed", "scope": "host"}],
              "required_evidence": [], "allowed_torch_calls": []}
    res = check_compliance("sorted = None\n", "triton", rules2)
    assert res.verdict == "audit_only"


def test_config_record_dict_is_not_a_cache_but_data_ptr_keys_are():
    assert analyze("_LAST_CONFIG = {}\ndef run(x):\n    _LAST_CONFIG.update({'BLOCK': 64})\n    return x\n", "triton").verdict() == "clear"
    assert analyze("_cache = {}\ndef run(x):\n    return _cache.get(x.data_ptr())\n", "triton").verdict() == "audit_only"
    assert analyze("_cache = {}\ndef run(x):\n    _cache[x.data_ptr()] = 1\n    return x\n", "triton").verdict() == "review_required"


# ---------------------------------------------------------------- R2 / R9: job schema, timing normalization
def test_timing_settings_normalize_study_keys(study):
    t = timing_settings(study, run_type="validation")
    assert t == {"warmup": 1, "repeat": 3, "use_cuda_graph": True, "flush": True,
                 "capture_failure_policy": "time_eagerly_and_flag", "record_prep_runs": True}
    assert timing_settings(study)["capture_failure_policy"] == "timing_error"          # formal (frozen 2026-10-06)


def test_evaluation_job_carries_tolerance_rules_and_timing(study):
    job = build_evaluation_job(operator="vector_add", dtype="fp16", params={"n": 16}, dsl="triton", device="B200",
                               arch=None, rules=VA_RULES, study=study, identity={"trajectory_id": "t"})
    w = job.worker_job(source_path="/x/impl_triton.py", sandbox_dir="/x", seed=3, round_index=2, attempt=1)
    assert validate_worker_job(w) == [] and w["atol"] == job.atol and w["rules"] is VA_RULES
    assert w["expected_timing_mode"] == "graph" and w["identity"]["round"] == 2
    rec = job.record()
    assert "rules" not in rec and len(rec["rules_sha256"]) == 64
    bad = dict(w); bad.pop("timing")
    assert any("timing" in x for x in validate_worker_job(bad))


def test_prompt_and_evaluator_tolerance_agree(study, folds):
    ctx, job, _ = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    assert (ctx.atol, ctx.rtol) == (job.atol, job.rtol)


# ---------------------------------------------------------------- R4: worker interface / autotuner / config checks (CPU)
@pytest.fixture
def cpu_worker(monkeypatch, tmp_path):
    from tilebench.llm.v2.evaluation import worker as w
    from tilebench.llm.v2.evaluation.timing import TimingRecord
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    import tilebench.data.tensors as tensors
    monkeypatch.setattr(tensors, "get_generator", lambda op: (lambda n, dtype: (torch.randn(n, dtype=dtype), torch.randn(n, dtype=dtype))))
    import tilebench.llm.v2.evaluation.timing as timing

    def fake_measure(f, **kw):
        f()
        return TimingRecord(1, 3, [0.5, 0.5, 0.5], 0.5, True, True, True, None, 4, "graph", True, 253)
    monkeypatch.setattr(timing, "measure", fake_measure)

    def run(src: str, cases: list | None = None) -> dict:
        p = tmp_path / "impl_triton.py"
        p.write_text(src)
        study = ms.load_study()
        job = build_evaluation_job(operator="vector_add", dtype="fp32", cases=cases, params=None if cases else {"n": 64},
                                   dsl="triton", device="B200", arch=None, rules=VA_RULES, study=study)
        return w.run_job(job.worker_job(source_path=str(p), sandbox_dir=str(tmp_path), seed=1, round_index=1, attempt=1))
    return run


def test_worker_requires_exports_and_fixed_config(cpu_worker):
    ok = cpu_worker("import torch\ndef run(x, y):\n    return x + y\ndef get_last_config():\n    return {'BLOCK': 64}\n")
    assert ok["status"] == "valid" and ok["cases"][0]["config"] == {"BLOCK": 64} and ok["valid_cases"] == ok["cases_total"] == 1
    assert ok["cases"][0]["latency_ms_samples"] == [0.5, 0.5, 0.5] and ok["timing_mode_differs"] is False
    assert abs(ok["latency_ms_geomean"] - 0.5) < 1e-12 and len(ok["cases"][0]["config_reads"]) == 3
    missing = cpu_worker("import torch\ndef run(x, y):\n    return x + y\n")
    assert missing["status"] == "interface_error" and "get_last_config" in missing["diagnostic"]
    notdict = cpu_worker("import torch\ndef run(x, y):\n    return x + y\ndef get_last_config():\n    return [1]\n")
    assert notdict["status"] == "interface_error"
    drift = cpu_worker("import torch\n_n = [0]\ndef run(x, y):\n    _n[0] += 1\n    return x + y\ndef get_last_config():\n    return {'calls': _n[0]}\n")
    assert drift["status"] == "interface_error" and "changed between calls" in drift["diagnostic"]


def test_worker_detects_autotuner_object_and_output_aliasing(cpu_worker):
    tuned = ("import torch, triton\nimport triton.language as tl\n"
             "@triton.autotune(configs=[triton.Config({'B': 1})], key=[])\n@triton.jit\ndef k(p): pass\n"
             "def run(x, y):\n    return x + y\ndef get_last_config():\n    return {}\n")
    res = cpu_worker(tuned)
    assert res["status"] == "contract_violation" and "Autotuner" in res["diagnostic"]
    alias = "import torch\ndef run(x, y):\n    x.add_(y)\n    return x\ndef get_last_config():\n    return {}\n"
    res2 = cpu_worker(alias)
    assert res2["status"] == "contract_violation" and "shares storage" in res2["diagnostic"]


# ---------------------------------------------------------------- R11: oracle aliasing
def test_reference_returning_mutated_input_is_frozen_before_restore():
    def ref(x):
        x.add_(1)
        return x

    def cheat(x):           # returns the input untouched: must NOT pass
        return x
    res = anticache.run_numerical_checks(cheat, ref, lambda: (torch.zeros(8),), atol=0, rtol=0,
                                         mutable_indices={0}, sync=lambda: None, aliasing_allowed=True)
    assert not res[0].ok

    def honest(x):
        x.add_(1)
        return x.clone()
    res2 = anticache.run_numerical_checks(honest, ref, lambda: (torch.zeros(8),), atol=0, rtol=0,
                                          mutable_indices={0}, sync=lambda: None)
    assert [r.ok for r in res2] == [True, True, True] and res2[0].inputs_mutated == [0]


# ---------------------------------------------------------------- R5: providers / factory
def test_provider_kwargs_carry_the_configured_settings():
    from tilebench.llm.v2.providers import anthropic_messages as am, openai_responses as oa
    from tilebench.llm.v2.providers.factory import generator_spec
    models = ms.load_models()
    g = generator_spec(models, "gpt", accept_status=("approved", "candidate"))
    kw = oa.build_kwargs(GenerationRequest(system="s", user="u", model_id=g.model_id, provider="openai", settings=g.settings))
    assert kw["model"] == "gpt-6.1-sol" and kw["reasoning"] == {"effort": "xhigh"} and kw["max_output_tokens"] == 128000
    c = generator_spec(models, "claude", accept_status=("approved", "candidate"))
    kw2 = am.build_kwargs(GenerationRequest(system="s", user="u", model_id=c.model_id, provider="anthropic", settings=c.settings))
    assert kw2["model"] == "claude-opus-5-5" and kw2["max_tokens"] == 128000
    assert kw2["thinking"] == {"type": "adaptive"} and kw2["output_config"] == {"effort": "xhigh"}
    assert c.api_key_env == "CLAUDE_API_KEY" and g.api_key_env == "OPENAI_API_KEY"
    assert generator_spec(models, "gpt").status == "approved"      # owner-approved: formal runs accept it
    with pytest.raises(ms.ManifestError):
        generator_spec(models, "gpt", accept_status=("candidate",))


def test_factory_refuses_without_key_env(monkeypatch):
    from tilebench.llm.v2.providers.factory import build_provider, generator_spec
    spec = generator_spec(ms.load_models(), "claude")
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-used")
    with pytest.raises(RuntimeError, match="CLAUDE_API_KEY"):
        build_provider(spec, timeout_s=1)


def test_sdk_exceptions_are_classified():
    import httpx
    import anthropic
    import openai
    from tilebench.llm.v2.providers import anthropic_messages as am, openai_responses as oa
    req = httpx.Request("POST", "https://x")
    bad = openai.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)
    assert isinstance(oa.classify_exception(bad), ProviderConfigError)
    rate = openai.RateLimitError("rl", response=httpx.Response(429, request=req), body=None)
    assert oa.classify_exception(rate).charged == "no"
    assert oa.classify_exception(openai.APITimeoutError(request=req)).charged == "unknown"
    auth = anthropic.AuthenticationError("a", response=httpx.Response(401, request=req), body=None)
    assert am.classify_exception(auth).kind == "auth"
    assert am.classify_exception(anthropic.APIConnectionError(request=req)).charged == "no"


# ---------------------------------------------------------------- R6: cost / ledger
def _ctx(study, folds):
    ctx, job, rules = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    e = eligibility("vector_add", "fp16", "B200", "triton", study, folds)
    st = new_trajectory_state(e, ctx, "m", "base", "h", "c", "t", run_type="validation", campaign="t")
    cfg = RunnerConfig(model_id="m", provider_name="mock", settings={}, feedback_limits=study["feedback"], retry_backoff_s=0.0)
    return ctx, job, st, cfg


def test_attempt_cost_rules():
    usage = normalize_openai_responses({"input_tokens": 10, "output_tokens": 5}).to_dict()
    assert attempt_cost([{"outcome": "succeeded", "charged": "known"}], usage) == (15, "known")
    assert attempt_cost([{"outcome": "failed", "charged": "no"}, {"outcome": "succeeded"}], usage) == (15, "known")
    assert attempt_cost([{"outcome": "failed", "charged": "unknown"}, {"outcome": "succeeded"}], usage) == (None, "unknown")
    assert attempt_cost([], None) == (0, "not_sent")


def test_unknown_transport_charge_propagates_to_the_attempt(tmp_path, study, folds):
    ctx, job, st, cfg = _ctx(study, folds)
    text = scripted_text("impl_triton.py", "# MOCK: valid 1.0\ndef run(*a): pass\ndef get_last_config(): return {}")
    prov = MockProvider([{"text": text, "transport_failures": 1, "transport_charged": "unknown"}])
    r = TrajectoryRunner(state=st, tdir=tmp_path / "t", ctx=ctx, provider=prov, evaluator=MockEvaluator(), job=job, cfg=cfg, sleep=lambda s: None)
    r.step()
    att = st.rounds[0].attempts[0]
    assert att.cost is None and att.cost_status == "unknown" and len(att.transport) == 2
    assert att.transport[0]["charged"] == "unknown" and att.transport[1]["outcome"] == "succeeded"
    rows = read_jsonl(tmp_path / "t" / "transport.jsonl")
    assert [x["event"] for x in rows] == ["sending", "failed", "sending", "succeeded"]


def test_prompt_too_long_is_recorded_as_zero_cost_not_unknown(tmp_path, study, folds):
    ctx, job, st, cfg = _ctx(study, folds)
    cfg.total_prompt_max_chars = 10
    r = TrajectoryRunner(state=st, tdir=tmp_path / "t", ctx=ctx, provider=MockProvider([]), evaluator=MockEvaluator(), job=job, cfg=cfg)
    r.step()
    att = st.rounds[0].attempts[0]
    assert att.cost == 0 and att.cost_status == "not_sent" and att.transport_attempts == 0
    rows = read_jsonl(tmp_path / "t" / "usage.jsonl")
    assert rows[0]["usage"]["status"] == "not_sent" and rows[0]["usage"]["logical_total"] == 0


def test_crash_between_response_and_ledger_is_reconciled_on_resume(tmp_path, study, folds):
    ctx, job, st, cfg = _ctx(study, folds)
    text = scripted_text("impl_triton.py", "# MOCK: valid 1.0\ndef run(*a): pass\ndef get_last_config(): return {}")
    tdir = tmp_path / "t"
    r = TrajectoryRunner(state=st, tdir=tdir, ctx=ctx, provider=MockProvider([{"text": text}]), evaluator=MockEvaluator(), job=job, cfg=cfg)
    r.step()
    (tdir / "usage.jsonl").unlink()                       # simulate the crash window
    # resume from a state that does not yet know the attempt (as if the process died before saving it)
    st2 = new_trajectory_state(eligibility("vector_add", "fp16", "B200", "triton", study, folds), ctx, "m", "base", "h", "c", "t",
                               run_type="validation", campaign="t")
    r2 = TrajectoryRunner(state=st2, tdir=tdir, ctx=ctx, provider=MockProvider([]), evaluator=MockEvaluator(), job=job, cfg=cfg)
    r2.step()
    rows = read_jsonl(tdir / "usage.jsonl")
    assert len(rows) == 1 and rows[0]["reconciled"] is True and st2.rounds[0].attempts[0].cost == st.rounds[0].attempts[0].cost
    assert any("reconciled" in n for n in st2.notes)


def test_orphaned_sending_event_makes_the_attempt_cost_unknown(tmp_path, study, folds):
    ctx, job, st, cfg = _ctx(study, folds)
    tdir = tmp_path / "t"
    from tilebench.llm.v2.providers.ledger import append_jsonl
    append_jsonl(tdir / "transport.jsonl", {"round": 1, "attempt": 1, "transport_attempt": 1, "event": "sending"})
    text = scripted_text("impl_triton.py", "# MOCK: valid 1.0\ndef run(*a): pass\ndef get_last_config(): return {}")
    r = TrajectoryRunner(state=st, tdir=tdir, ctx=ctx, provider=MockProvider([{"text": text}]), evaluator=MockEvaluator(), job=job, cfg=cfg)
    r.step()
    att = st.rounds[0].attempts[0]
    assert att.cost is None and att.transport[0]["outcome"] == "orphaned" and att.transport[1]["transport_attempt"] == 2


def test_provider_refusal_stops_without_retry(tmp_path, study, folds):
    ctx, job, st, cfg = _ctx(study, folds)

    class Refusing:
        name = "mock"; usage_schema = "openai_responses_v1"; calls = 0

        def generate(self, req):
            self.calls += 1
            raise ProviderConfigError("BadRequestError (400): unsupported parameter", status_code=400)
    p = Refusing()
    r = TrajectoryRunner(state=st, tdir=tmp_path / "t", ctx=ctx, provider=p, evaluator=MockEvaluator(), job=job, cfg=cfg, sleep=lambda s: None)
    assert r.run() == "incomplete" and p.calls == 1
    assert "provider refused" in st.stop_reason and st.rounds[0].attempts[0].verdict == "provider_refused"


def test_truncated_response_is_a_format_error(tmp_path, study, folds):
    ctx, job, st, cfg = _ctx(study, folds)

    class Trunc:
        name = "mock"; usage_schema = "openai_responses_v1"

        def generate(self, req):
            return GenerationResult(text="```python title=\"impl_triton.py\"\ndef run(): pass\n```", model_id="m", provider="mock",
                                    response_id="r", usage_raw=None, usage=normalize_openai_responses({"input_tokens": 1, "output_tokens": 1}),
                                    transport_attempts=1, elapsed_s=0.0, terminal_status="incomplete:max_output_tokens", truncated=True)
    r = TrajectoryRunner(state=st, tdir=tmp_path / "t", ctx=ctx, provider=Trunc(), evaluator=MockEvaluator(), job=job, cfg=cfg)
    r.step()
    assert st.rounds[0].status == "format_error" and "truncated" in st.rounds[0].diagnostic


# ---------------------------------------------------------------- execution-confirmed violation + fallback prompt (item 11)
def test_execution_confirmed_violation_closes_the_round_and_next_round_hides_its_code(tmp_path, study, folds):
    # protocol revision 3: an execution-confirmed violation closes the round with its ONE attempt (no regeneration);
    # the next round gets the diagnostic and the last compliant code, never the rejected code
    ctx, job, st, cfg = _ctx(study, folds)
    viol = scripted_text("impl_triton.py", "# MOCK: contract_violation\n# SECRET_REJECTED_CODE\ndef run(*a): pass\ndef get_last_config(): return {}")
    good = scripted_text("impl_triton.py", "# MOCK: valid 1.0\n# GOOD_CODE\ndef run(*a): pass\ndef get_last_config(): return {}")
    prov = MockProvider([{"text": good}, {"text": viol}, {"text": good}])
    r = TrajectoryRunner(state=st, tdir=tmp_path / "t", ctx=ctx, provider=prov, evaluator=MockEvaluator(), job=job, cfg=cfg)
    for _ in range(2):
        r.step()                                           # round 1 valid
    for _ in range(2):
        r.step()                                           # round 2: one generation, execution-confirmed violation
    assert st.rounds[1].status == "contract_violation" and [a.verdict for a in st.rounds[1].attempts] == ["confirmed_violation"]
    assert len(prov.requests) == 2
    r.step()                                               # round 3 request
    req = prov.requests[-1].user
    assert len(prov.requests) == 3 and "Optimization round 3 of 5" in req
    assert "SECRET_REJECTED_CODE" not in req and "GOOD_CODE" in req and "no compliant implementation" in req


def test_valid_round_needs_consistent_finite_samples():
    st = TrajectoryState(schema="s", trajectory_id="t", task={}, model="m", condition="base", config_hash="h", content_hashes={}, output_file="f")
    sm.apply_attempt(st, 1, AttemptRecord(1, "initial", "r", 1, 1, "clear", source_path="/dev/null"))
    sm.apply_evaluation(st, 1, {"status": "valid", "latency_ms_mean": 2.0, "latency_ms_samples": [1.0, 1.0, 1.0]})
    assert st.rounds[0].status == "timing_error"
    sm.apply_attempt(st, 2, AttemptRecord(1, "initial", "r", 1, 1, "clear", source_path="/dev/null"))
    sm.apply_evaluation(st, 2, {"status": "valid", "latency_ms_mean": float("nan"), "latency_ms_samples": [float("nan")] * 3})
    assert st.rounds[1].status == "timing_error"


# ---------------------------------------------------------------- R7: metrics
def rounds(*specs):
    return [{"round": i, "attempts": [{"cost": c} for c in costs], "valid": v, "latency_ms": lat}
            for i, (costs, v, lat) in enumerate(specs, 1)]


def test_unknown_cost_and_missing_sol_are_not_zero():
    unknown = e.curve(1.0, rounds(([None], True, 1.0)))
    assert e.efficiency_at(unknown, 100) is None
    nosol = e.curve(None, rounds(([10], True, 1.0)))
    assert nosol.status == "sol_unavailable" and e.efficiency_at(nosol, 100) is None and nosol.best_efficiency is None
    agg = e.aggregate({("op", "fp16"): unknown, ("op2", "fp16"): e.curve(1.0, rounds(([10], True, 2.0)))}, [100],
                      {("op", "fp16"): "eligible", ("op2", "fp16"): "eligible"})
    assert agg["mean"] == [None] and agg["lower_bound_mean"] == [0.25] and agg["partial"] and agg["undetermined"]
    real_zero = e.aggregate({("op2", "fp16"): e.curve(1.0, rounds(([10], False, None)))}, [100], {("op2", "fp16"): "eligible"})
    assert real_zero["mean"] == [0.0] and not real_zero["partial"]
    missing = e.aggregate({}, [100], {("op3", "fp16"): "eligible"})
    assert missing["mean"] == [None] and missing["missing"] == [("op3", "fp16")]


def test_non_finite_latency_is_not_valid_and_pairing_needs_same_coverage():
    c = e.curve(1.0, rounds(([10], True, float("nan"))))
    assert not c.points[0].valid and c.audit_flags
    a = e.aggregate({("op", "fp16"): e.curve(1.0, rounds(([10], True, 1.0)))}, [10], {("op", "fp16"): "eligible"})
    b = e.aggregate({("op", "fp32"): e.curve(1.0, rounds(([10], True, 1.0)))}, [10], {("op", "fp32"): "eligible"})
    with pytest.raises(ValueError, match="coverage"):
        e.paired_difference(a, b)


# ---------------------------------------------------------------- R8: distillation binding
def test_distillation_refuses_identity_mismatch_escape_and_validation_runs(tmp_path, study, folds):
    from tilebench.llm.v2.distillation import access
    from tilebench.llm.v2.distillation.orchestrator import DistillerConfig, extract_observations, verify_state_identity
    out = run_two_mock_trajectories(tmp_path, study, folds)
    index = access.build_index(out)
    assert all(r.state_sha256 for r in index) and all(r.run_type == "validation" for r in index)
    scope = access.evaluation_scope(index, dsl="triton", held_out_fold="A", study=study, root=out)
    assert not scope.selected and any("validation" in k for k in scope.excluded)   # validation runs are never sources
    # forge a formal reference: identity is re-checked against the state file
    ref = index[0]
    forged = access.TrajectoryRef(**{**ref.__dict__, "run_type": "formal"})
    scope2 = access.evaluation_scope([forged], dsl="triton", held_out_fold="A", study=study, root=out)
    assert scope2.selected
    with pytest.raises(access.EvidenceAccessError, match="identity"):
        verify_state_identity(forged, json.loads(Path(forged.path).read_text()))
    outside = access.TrajectoryRef(**{**forged.__dict__, "path": str(tmp_path / "elsewhere.json")})
    scope3 = access.evaluation_scope([outside], dsl="triton", held_out_fold="A", study=study, root=out / "sub")
    with pytest.raises(access.EvidenceAccessError, match="outside"):
        scope3.open(outside)
    tampered = access.TrajectoryRef(**{**forged.__dict__, "state_sha256": "0" * 64})
    with pytest.raises(access.EvidenceAccessError, match="hash"):
        access.evaluation_scope([tampered], dsl="triton", held_out_fold="A", study=study, root=out).open(tampered)
    cfg = DistillerConfig(model_id="m", provider_name="mock", settings={}, dsl_version="3.6.0")
    with pytest.raises(access.EvidenceAccessError):
        extract_observations(scope2, MockProvider([{"text": "[]"}]), cfg, read_state=lambda p: json.loads(p.read_text()))


def run_two_mock_trajectories(tmp_path, study, folds):
    from tilebench.llm.v2.devtools import run_mock_trajectory
    out = tmp_path / "camp"
    run_mock_trajectory(out, operator="layernorm", dtype="fp16", device="B200", dsl="triton", condition="base")
    return out


def test_distillation_persists_and_reuses_observations(tmp_path, study, folds):
    from tilebench.llm.v2.distillation import access
    from tilebench.llm.v2.distillation.fixtures import synthetic_index
    from tilebench.llm.v2.distillation.orchestrator import DistillerConfig, extract_observations, synthesize
    index = synthetic_index(tmp_path / "idx", folds, dsl="triton", devices=["B200"], models=["gpt"], operators=["layernorm", "softmax"])
    scope = access.evaluation_scope(index, dsl="triton", held_out_fold="A", study=study, root=tmp_path / "idx")
    cfg = DistillerConfig(model_id="m", provider_name="mock", settings={}, dsl_version="3.6.0")
    prov = MockProvider([{"text": "[obs]"}] * 2)
    obs = extract_observations(scope, prov, cfg, read_state=lambda p: json.loads(p.read_text()), out_dir=tmp_path / "d")
    assert len(obs) == 2 and len(prov.requests) == 2 and (tmp_path / "d" / "observations").exists()
    again = extract_observations(scope, MockProvider([]), cfg, read_state=lambda p: json.loads(p.read_text()), out_dir=tmp_path / "d")
    assert [o["trajectory_id"] for o in again] == [o["trajectory_id"] for o in obs]      # no new request
    other = access.evaluation_scope(index, dsl="triton", held_out_fold="B", study=study, root=tmp_path / "idx")
    with pytest.raises(access.EvidenceAccessError, match="another scope"):
        synthesize(other, obs, MockProvider([{"text": "skill"}]), cfg, models=["gpt"])
    h1 = cfg.config_hash()
    import tilebench.llm.v2.distillation.orchestrator as orch
    orig = orch.load_template
    orch.load_template = lambda n: orig(n) + "\nchanged"
    try:
        assert cfg.config_hash() != h1                       # template CONTENT is part of the hash
    finally:
        orch.load_template = orig


# ---------------------------------------------------------------- R9: grants
def test_provider_grants_are_separate_from_status(skill_env, study):
    from tilebench.llm.v2.skills import loader
    m, root = skill_env
    entry = m["reference"]["triton"]["3.6.0"]
    assert entry["sendable_to"] == [] and entry["publishable"] is False      # register_asset grants nothing by default
    with pytest.raises(loader.SkillPermissionError, match="not granted"):
        loader.load_component(m, "reference", "triton", "3.6.0", provider="openai")
    loader.load_component(m, "reference", "triton", "3.6.0")                 # no provider named: status check only
    entry["sendable_to"] = ["openai"]
    loader.load_component(m, "reference", "triton", "3.6.0", provider="openai")
    with pytest.raises(loader.SkillPermissionError):
        loader.load_component(m, "reference", "triton", "3.6.0", provider="anthropic")
    m["reference"]["nki"]["beta5"]["sendable_to"] = ["openai"]              # private can never be granted
    assert loader.sendable_to(m["reference"]["nki"]["beta5"]) == []


def test_real_manifest_grants_match_permissions():
    from tilebench.llm.v2.skills.loader import load_manifest
    m = load_manifest()
    for kind in ("reference", "device"):
        for key, versions in m[kind].items():
            for ver, en in versions.items():
                if en["permission"] == "public":
                    assert set(en["sendable_to"]) == {"openai", "anthropic"} and en["publishable"]
                else:
                    assert en["sendable_to"] == [] and not en["publishable"]


# ---------------------------------------------------------------- R12: isolation / publication
def test_isolation_report_and_bwrap_argv(tmp_path):
    assert detect_isolation("none")["backend"] == "none"
    argv = _bwrap_base(tmp_path)
    assert "--unshare-net" in argv and "--tmpfs" in argv and "/home" in argv and str(tmp_path) in argv and "--ro-bind" in argv


def test_publication_withholds_non_publishable_context(tmp_path, study, folds):
    from tilebench.llm.v2.orchestration.publication import export_publication
    out = run_two_mock_trajectories(tmp_path, study, folds)
    st = TrajectoryState.load(next(out.rglob("trajectory.json")))
    key = next(k for k in st.content_hashes if k.startswith("reference:"))
    kind, rest = key.split(":", 1)
    name, _, ver = rest.rpartition("@")
    manifest = {kind: {name: {ver: {"permission": "public", "publishable": True, "sendable_to": ["openai"]}}},
                "device": {}}
    dev = next(k for k in st.content_hashes if k.startswith("device:"))
    dname, _, dver = dev.split(":", 1)[1].rpartition("@")
    manifest["device"][dname] = {dver: {"permission": "public", "publishable": True}}
    res = export_publication(out / "mock", tmp_path / "pub", manifest=manifest)
    assert res["trajectories"] == 1 and res["withheld"] == 0 and (tmp_path / "pub" / "INDEX.json").exists()
    manifest["device"][dname][dver]["publishable"] = False
    res2 = export_publication(out / "mock", tmp_path / "pub2", manifest=manifest)
    assert res2["trajectories"] == 0 and res2["withheld"] == 1
    red = json.loads((tmp_path / "pub2" / "REDACTIONS.json").read_text())
    assert red["withheld_trajectories"][0]["reason"].startswith("component device:")


def test_required_evidence_sees_string_literals_but_forbidden_rules_do_not():
    rules = {"forbidden_substitutions": [{"pattern": r"torch\.softmax\(", "message": "m", "level": "confirmed", "scope": "host"}],
             "required_evidence": [{"any_of": [r"-inf", r"float\(['\"]-inf['\"]\)"], "message": "no -inf evidence", "level": "suspicious"}],
             "allowed_torch_calls": []}
    src = "import triton.language as tl\ndef run(x):\n    m = float('-inf')\n    s = 'torch.softmax('\n    return x\n"
    res = check_compliance(src, "triton", rules)
    assert res.verdict == "clear", res.to_dict()          # literal -inf counts; literal torch.softmax( does not


def test_cutile_raw_memory_loads_count_as_kernel_evidence():
    rules = json.loads(Path("tilebench/llm/v2/contracts/data/vector_add/evaluator_rules.json").read_text())
    src = ("import torch\nimport cuda.tile as ct\n\n@ct.kernel\ndef k(x, y, out, N: ct.Constant[int]):\n"
           "    offs = ct.bid(0) * 8192 + ct.arange(8192, dtype=ct.int32)\n    mask = offs < N\n"
           "    a = x.get_raw_memory().load_offset(offs, mask=mask, padding_value=0)\n"
           "    b = y.get_raw_memory().load_offset(offs, mask=mask, padding_value=0)\n"
           "    out.get_raw_memory().store_offset(offs, a + b, mask=mask)\n\n"
           "def run(x, y):\n    out = torch.empty_like(x)\n    ct.launch(torch.cuda.current_stream(), (ct.cdiv(x.numel(), 8192),), k, (x, y, out, x.numel()))\n    return out\n\n"
           "def get_last_config():\n    return {'tile': 8192}\n")
    assert check_compliance(src, "cutile", rules).verdict == "clear"


def test_worker_hang_after_candidate_load_is_the_candidates_failure(tmp_path):
    from tilebench.llm.v2.evaluation.launcher import classify_no_result
    (tmp_path / "progress.json").write_text(json.dumps({"phase": "candidate_loaded", "t": 0}))
    r = classify_no_result(tmp_path, timed_out=True, timeout_s=10, rc=None, stderr="")
    assert r["status"] == "runtime_error" and "candidate exceeded" in r["diagnostic"] and r["worker_phase"] == "candidate_loaded"
    (tmp_path / "progress.json").write_text(json.dumps({"phase": "worker_started", "t": 0}))
    r2 = classify_no_result(tmp_path, timed_out=False, timeout_s=10, rc=137, stderr="killed")
    assert r2["status"] == "infrastructure_incomplete" and "before the candidate was loaded" in r2["diagnostic"]
    (tmp_path / "progress.json").unlink()
    assert classify_no_result(tmp_path, timed_out=True, timeout_s=10, rc=None, stderr="")["status"] == "infrastructure_incomplete"


def test_retry_incomplete_reopens_only_evaluation_side_failures():
    from tilebench.llm.v2.orchestration.campaign import retry_incomplete
    st = TrajectoryState(schema="s", trajectory_id="t", task={}, model="m", condition="base", config_hash="h", content_hashes={}, output_file="f")
    sm.apply_attempt(st, 1, AttemptRecord(1, "initial", "r", 1, 1, "clear", source_path="/dev/null"))
    sm.apply_evaluation(st, 1, {"status": "infrastructure_incomplete", "diagnostic": "worker exceeded 10s before the candidate was loaded"})
    assert st.status == "incomplete"
    assert retry_incomplete(st) and st.status == "in_progress" and st.rounds[0].status == "pending"
    assert sm.next_action(st).kind == "evaluate" and st.notes
    st2 = TrajectoryState(schema="s", trajectory_id="t", task={}, model="m", condition="base", config_hash="h", content_hashes={}, output_file="f")
    sm.apply_attempt(st2, 1, AttemptRecord(1, "initial", "r", 1, 2, "transport_failed", cost=None, cost_status="unknown"))
    assert st2.status == "incomplete" and retry_incomplete(st2) is None and st2.status == "incomplete"


def test_data_ptr_guard_is_not_a_cache_key_but_dict_use_is():
    guard = "import torch\ndef run(x):\n    y = x.contiguous()\n    assert y.data_ptr() == x.data_ptr()\n    return y.clone()\n"
    assert not [e for e in analyze(guard, "triton").evidence if e.category == "cache" and e.level != "audit"]
    keyed = "import torch\n_seen = {}\ndef run(x):\n    k = x.data_ptr()\n    if k in _seen:\n        return _seen[k]\n    return x\n"
    assert [e for e in analyze(keyed, "triton").evidence if e.category == "cache"]
    sub = "import torch\n_c = {}\ndef run(x):\n    return _c[x.data_ptr()]\n"
    assert any(e.line == 4 for e in analyze(sub, "triton").evidence if e.category == "cache")
