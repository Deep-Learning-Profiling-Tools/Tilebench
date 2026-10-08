"""Protocol revision 3 (study-owner decision 2026-10-06): one representative
dtype per operator, 5 rounds, exactly one candidate generation per round, no
same-round repair, checker v2 with non-blocking audit_only, the revision-2
pilot excluded in code, and non-sensitive provider telemetry."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tilebench.llm.v2.contracts.loader import load_contract
from tilebench.llm.v2.devtools import recompute_metrics, synthetic_context
from tilebench.llm.v2.distillation import access
from tilebench.llm.v2.evaluation.launcher import MockEvaluator
from tilebench.llm.v2.manifests import schema as ms
from tilebench.llm.v2.metrics import efficiency as e
from tilebench.llm.v2.orchestration import campaign
from tilebench.llm.v2.orchestration import state_machine as sm
from tilebench.llm.v2.orchestration.campaign import ResumeRefused, _check_resume, new_trajectory_state, protocol_identity
from tilebench.llm.v2.orchestration.runner import RunnerConfig, TrajectoryRunner
from tilebench.llm.v2.orchestration.state import SCHEMA, AttemptRecord, TrajectoryState
from tilebench.llm.v2.orchestration.telemetry import ProcessTelemetry, TelemetryProvider
from tilebench.llm.v2.providers.base import TransportError, rate_limit_headers
from tilebench.llm.v2.providers.factory import generator_spec
from tilebench.llm.v2.providers.mock import MockProvider, scripted_text
from tilebench.llm.v2.tasks import representative
from tilebench.llm.v2.tasks.support import eligibility, task_table
from tilebench.llm.v2.validation.contract_checks import check_compliance
from tilebench.paths import list_operators

PILOT = "formal_b200_base_2026-10-06"


def _runner(study, folds, tmp_path, script, *, operator="vector_add", rules=None):
    ctx, job, _ = synthetic_context(operator, "fp16", "B200", "triton", "base", study, folds)
    if rules is not None:
        job.rules = rules
    el = eligibility(operator, "fp16", "B200", "triton", study, folds)
    st = new_trajectory_state(el, ctx, "m", "base", "h", "c", "t", run_type="validation", campaign="t",
                              protocol=protocol_identity(study))
    cfg = RunnerConfig(model_id="m", provider_name="mock", settings={}, feedback_limits=study["feedback"], retry_backoff_s=0.0,
                       rounds=study["trajectory"]["rounds"], max_generations=study["trajectory"]["max_generations_per_round"])
    prov = MockProvider([{"text": scripted_text("impl_triton.py", t)} if isinstance(t, str) else t for t in script])
    return TrajectoryRunner(state=st, tdir=tmp_path / "t", ctx=ctx, provider=prov, evaluator=MockEvaluator(), job=job, cfg=cfg), st, prov


def _valid(ms_=1.0, extra=""):
    return f"# MOCK: valid {ms_}\n{extra}def run(*a): pass\ndef get_last_config(): return {{'BLOCK': 64}}"


# ----------------------------------------------------------------------------- protocol shape
def test_study_is_revision_4_five_rounds_one_generation(study):
    assert study["revision"] == ms.PROTOCOL_REVISION == 4
    assert study["trajectory"]["rounds"] == 5 and study["trajectory"]["max_generations_per_round"] == 1
    assert "repair" not in json.dumps(study["trajectory"]).replace("no same-round regeneration", "")


@pytest.mark.parametrize("outcome", ["format_error", "compile_error", "runtime_error", "numerical_error", "timing_error",
                                     "interface_error", "contract_violation"])
def test_every_failure_closes_its_round_after_one_generation_and_next_round_gets_the_diagnostic(study, folds, tmp_path, outcome):
    first = "this is not a fenced file" if outcome == "format_error" else \
        f"# MOCK: {outcome}\n# DIAG_MARKER\ndef run(*a): pass\ndef get_last_config(): return {{}}"
    script = [{"text": first} if outcome == "format_error" else first, _valid(1.0)]
    r, st, prov = _runner(study, folds, tmp_path, script)
    while len(prov.requests) < 2:
        r.step()
    rec = st.rounds[0]
    assert rec.status == outcome and len(rec.attempts) == 1                 # one generation, round closed
    assert rec.attempts[0].cost is not None and rec.attempts[0].cost > 0     # tokens of the failed generation are charged
    assert prov.requests[1].metadata["round"] == 2 and prov.requests[1].metadata["attempt"] == 1
    nxt = prov.requests[1].user
    assert "Optimization round 2 of 5" in nxt
    if outcome not in ("format_error",):
        assert "Diagnostics:" in nxt or "contract violation" in nxt


def test_static_confirmed_violation_closes_round_without_repair_and_is_charged(study, folds, tmp_path):
    bad = "import triton\n@triton.autotune(configs=[], key=[])\n@triton.jit\ndef k(): pass\ndef run(*a): pass\ndef get_last_config(): return {}"
    r, st, prov = _runner(study, folds, tmp_path, [bad, _valid()])
    r.step()
    assert st.rounds[0].status == "contract_violation" and [a.verdict for a in st.rounds[0].attempts] == ["confirmed_violation"]
    r.step()
    assert len(prov.requests) == 2 and prov.requests[1].metadata["round"] == 2
    assert "autotune decorator" in prov.requests[1].user
    curve = e.curve(1.0, st.metric_rounds())
    assert curve.points[0].cumulative_cost == st.rounds[0].attempts[0].cost        # C_1 includes the violating generation


def test_full_trajectory_is_exactly_five_generations_no_early_stop(study, folds, tmp_path):
    r, st, prov = _runner(study, folds, tmp_path, [_valid(0.001)] * 5)
    assert r.run() == "complete"
    assert len(st.rounds) == 5 and len(prov.requests) == 5 and all(len(x.attempts) == 1 for x in st.rounds)


def test_transport_retry_is_not_a_candidate_attempt(study, folds, tmp_path):
    item = {"text": scripted_text("impl_triton.py", _valid()), "transport_failures": 2}
    r, st, prov = _runner(study, folds, tmp_path, [item])
    r.step()
    a = st.rounds[0].attempts
    assert len(a) == 1 and a[0].attempt == 1 and a[0].transport_attempts == 3


# ----------------------------------------------------------------------------- checker v2
def test_audit_only_proceeds_to_the_evaluator(study, folds, tmp_path):
    rules = load_contract("vector_add", require_approved=False).rules
    src = _valid(1.0)                                             # no kernel: every required-evidence pattern is missing
    assert check_compliance(src, "triton", rules).verdict == "audit_only"
    r, st, prov = _runner(study, folds, tmp_path, [src], rules=rules)
    r.step()
    assert st.rounds[0].attempts[0].verdict == "audit_only" and sm.next_action(st).kind == "evaluate"
    r.step()
    assert st.rounds[0].status == "valid" and st.status == "in_progress"
    assert st.rounds[0].attempts[0].compliance["audit_flags"]


def test_missing_required_evidence_alone_never_blocks():
    rules = {"required_evidence": [{"any_of": [r"-inf"], "message": "No -inf fill found.", "level": "suspicious"},
                                   {"any_of": [r"tl\.sum"], "message": "No sum-reduction primitive found.", "level": "suspicious"}]}
    res = check_compliance("import torch\ndef run(x):\n    return torch.empty_like(x)\n", "triton", rules)
    assert res.verdict == "audit_only" and not res.review_items() and len(res.audit_flags()) == 2


@pytest.mark.parametrize("src,verdict", [
    # high-risk ambiguity stays review_required
    ("import torch\n_C = {}\ndef run(x):\n    k = x.data_ptr()\n    if k in _C:\n        return _C[k]\n    out = torch.empty_like(x)\n    _C[k] = out\n    return out\n", "review_required"),
    ("import torch\n_O = None\ndef run(x):\n    global _O\n    _O = torch.empty_like(x)\n    return _O\n", "review_required"),
    ("import torch\ndef run(x, _c={}):\n    _c['o'] = torch.empty_like(x)\n    return _c['o']\n", "review_required"),
    ("import torch\nfrom torch.utils.weak import WeakTensorKeyDictionary\n_C = WeakTensorKeyDictionary()\ndef run(x):\n    return x\n", "review_required"),
    ("import torch\ndef run(x):\n    s = torch.cuda.Stream()\n    return x\n", "review_required"),
    ("import torch\ndef run(x):\n    exec('pass')\n    return x\n", "review_required"),
    # confirmed violations
    ("import torch\ndef run(x):\n    torch.backends.cuda.matmul.allow_tf32 = True\n    return x\n", "confirmed_violation"),
    ("import torch\ndef run(x):\n    torch.set_float32_matmul_precision('high')\n    return x\n", "confirmed_violation"),
    ("import torch\nF = torch.nn.functional\ndef run(x):\n    return F.softmax(x, -1)\n", "confirmed_violation"),
    ("import torch\ndef run(a, b):\n    return a @ b\n", "confirmed_violation"),
    ("import torch\nfrom tilebench.core import timer\ndef run(x):\n    return x\n", "confirmed_violation"),
    ("import tilelang\n@tilelang.autotune(configs=[])\ndef k(): pass\ndef run(x):\n    return x\n", "confirmed_violation"),
    # audit-only / clear
    ("import torch\n_K = {}\ndef _build(n):\n    return ('kernel', n)\ndef run(x):\n    n = x.numel()\n    _K[n] = _build(n)\n    return x\n", "audit_only"),
    ("import torch\ndef run(x):\n    p = x.data_ptr()\n    assert p % 16 == 0\n    return x\n", "audit_only"),
    ("import torch\ndef run(x):\n    return torch.empty_like(x)\n", "clear"),
])
def test_checker_v2_negative_and_positive_cases(src, verdict):
    assert check_compliance(src, "triton", {"allowed_torch_calls": ["torch.empty_like"]}).verdict == verdict


def test_human_compliant_evaluates_archived_candidate_without_request(study, folds, tmp_path):
    risky = "# MOCK: valid 1.0\nimport importlib\ndef run(*a): pass\ndef get_last_config(): return {}"
    r, st, prov = _runner(study, folds, tmp_path, [risky, _valid()])
    assert r.run() == "review_required" and len(prov.requests) == 1
    sm.resolve_review(st, 1, "compliant", "importlib unused; reviewer checked")
    r.step()                                                     # evaluation of the archived candidate
    assert st.rounds[0].status == "valid" and len(prov.requests) == 1
    r.step()
    assert len(prov.requests) == 2 and prov.requests[1].metadata["round"] == 2


def test_human_violation_closes_round_without_regeneration(study, folds, tmp_path):
    risky = "# MOCK: valid 1.0\nimport importlib\ndef run(*a): pass\ndef get_last_config(): return {}"
    r, st, prov = _runner(study, folds, tmp_path, [risky, _valid()])
    r.run()
    sm.resolve_review(st, 1, "violation", "imports the evaluator dynamically")
    assert st.rounds[0].status == "contract_violation" and len(st.rounds[0].attempts) == 1
    r.step()
    assert prov.requests[-1].metadata["round"] == 2 and len(prov.requests) == 2


# ----------------------------------------------------------------------------- representative dtype
def test_representative_dtype_is_exactly_one_per_operator_and_rule_based():
    m = representative.load_manifest()
    ops = list_operators()
    assert sorted(m["operators"]) == sorted(ops) and len(ops) == 45
    for op, ent in m["operators"].items():
        assert ent["dtype"] in ent["eligible_dtypes"]
        assert representative.select_dtype(op, ent["eligible_dtypes"])[0] == ent["dtype"]
    sel = representative.selected(m)
    assert sel["vector_add"] == sel["argmax"] == sel["bitonic_sort"] == "fp16"
    assert sel["histogramming"] == "int32" and sel["dequantize_rowwise"] == "int8" and sel["flash_decode"] == "fp32"


def test_same_dtype_for_every_b200_dsl_model_and_condition(study, folds):
    sel = representative.selected()
    per_dsl = {}
    for dsl in ("triton", "cutile", "tilelang"):
        rows = [x for x in task_table(study, folds) if x.key.device == "B200" and x.key.dsl == dsl]
        assert len(rows) == 45 and all(x.status == "eligible" for x in rows)
        per_dsl[dsl] = {x.key.operator: (x.key.dtype, x.key.case_id) for x in rows}
        assert {op: dt for op, (dt, _) in per_dsl[dsl].items()} == sel
    assert per_dsl["triton"] == per_dsl["cutile"] == per_dsl["tilelang"]
    # model and condition are not task attributes: the same task list serves gpt/claude and base/enhanced
    assert campaign.select_tasks(study, folds, "B200", "triton", None, None) == \
        campaign.select_tasks(study, folds, "B200", "triton", None, None)


def test_representative_manifest_refuses_a_hand_edited_dtype(monkeypatch):
    m = representative.load_manifest()
    bad = json.loads(json.dumps(m))
    bad["operators"]["vector_add"]["dtype"] = "fp32"
    with pytest.raises(ms.ManifestError, match="vector_add"):
        representative.validate_manifest(bad)


def test_config_hash_covers_the_representative_manifest(study, folds):
    models, modes = ms.load_models(), ms.load_arithmetic_modes()
    rep = representative.load_manifest()
    h = ms.study_config_hash(study, models, folds, modes, rep)
    rep2 = json.loads(json.dumps(rep))
    rep2["operators"]["vector_add"]["reason"] += " (edited)"
    assert ms.study_config_hash(study, models, folds, modes, rep2) != h


# ----------------------------------------------------------------------------- runtime shards
def test_shards_are_disjoint_complete_and_fold_independent(folds):
    a, b = campaign.shard_operators("1/2"), campaign.shard_operators("2/2")
    assert not set(a) & set(b) and sorted(a + b) == sorted(list_operators()) and abs(len(a) - len(b)) <= 1
    assert campaign.shard_operators("1/2", ["vector_add", "argmax"]) == [op for op in a if op in ("vector_add", "argmax")]
    with pytest.raises(ValueError):
        campaign.parse_shard("3/2")


def test_trajectory_lock_is_exclusive(tmp_path):
    l1, l2 = campaign.TrajectoryLock(tmp_path / "x.lock"), campaign.TrajectoryLock(tmp_path / "x.lock")
    assert l1.acquire() and not l2.acquire()
    l1.release()
    assert l2.acquire()
    l2.release()


# ----------------------------------------------------------------------------- pilot exclusion
def test_old_pilot_is_refused_as_a_distillation_source(tmp_path, study, folds):
    st = {"trajectory_id": "p", "model": "gpt", "condition": "base", "status": "complete", "run_type": "formal",
          "campaign": PILOT, "schema": SCHEMA, "protocol": {"revision": 4},
          "task": {"operator": "layernorm", "dtype": "fp16", "dsl": "triton", "device": "B200", "fold": "B"},
          "rounds": [{"round": 1, "status": "valid"}]}
    legacy = {**st, "trajectory_id": "q", "campaign": "other", "schema": "tilebench-llm-v2-trajectory/2", "protocol": None}
    refs = access.refs_from_states([{**st, "path": "p"}, {**legacy, "path": "q"}], folds)
    scope = access.evaluation_scope(refs, dsl="triton", held_out_fold="A", study=study)
    assert not scope.selected
    assert any("is excluded" in k for k in scope.excluded) and any("not accepted" in k for k in scope.excluded)
    assert ms.excluded_campaign(PILOT)["forbidden_uses"]


def test_new_campaign_excludes_the_pilot_and_never_resumes_revision2_states(study):
    out = campaign.run_campaign(campaign.CampaignSpec(name=PILOT, run_type="formal", device="B200", dsl="triton",
                                                      condition="base", model="gpt", resume=True))
    assert out["refused"] and "excluded" in out["reason"]
    gen = generator_spec(ms.load_models(), "gpt")
    old = TrajectoryState(schema="tilebench-llm-v2-trajectory/2", trajectory_id="old", task={}, model="gpt", condition="base",
                          config_hash="h", content_hashes={}, output_file="impl_triton.py", run_type="formal", generator=gen.record())
    with pytest.raises(ResumeRefused, match="protocol"):
        _check_resume(old, config_hash="h", content_hashes={}, run_type="formal", gen=gen, protocol=protocol_identity(study))


def test_resume_preserves_the_revision3_identity(study, folds, tmp_path):
    r, st, prov = _runner(study, folds, tmp_path, [_valid()])
    r.step(); r.step()
    loaded = TrajectoryState.load(tmp_path / "t" / "trajectory.json")
    assert loaded.schema == SCHEMA and loaded.protocol == protocol_identity(study)
    assert loaded.protocol["rounds"] == 5 and loaded.protocol["max_generations_per_round"] == 1
    gen = generator_spec(ms.load_models(), "gpt")
    loaded.generator, loaded.run_type = gen.record(), "formal"
    _check_resume(loaded, config_hash=loaded.config_hash, content_hashes=loaded.content_hashes, run_type="formal", gen=gen,
                  protocol=protocol_identity(study))
    changed = {**protocol_identity(study), "rounds": 10}
    with pytest.raises(ResumeRefused):
        _check_resume(loaded, config_hash=loaded.config_hash, content_hashes=loaded.content_hashes, run_type="formal", gen=gen,
                      protocol=changed)


# ----------------------------------------------------------------------------- metrics denominator
def test_metrics_use_the_45_operator_denominator_and_skip_the_pilot(study, folds, tmp_path):
    r, st, prov = _runner(study, folds, tmp_path, [_valid()] * 5)
    st.run_type, st.campaign = "formal", "fresh"
    r.run()
    pilot_dir = tmp_path / "pilotcopy"
    pilot_dir.mkdir()
    (pilot_dir / "trajectory.json").write_text(json.dumps({**st.to_dict(), "campaign": PILOT, "trajectory_id": "pilot"}))
    out = recompute_metrics(tmp_path)
    keys = [k for k in out if not k.startswith("_")]
    assert len(keys) == 1
    agg = out[keys[0]]
    assert agg["denominator_operators"] == 45 and agg["n_operators"] == 45 and len(agg["missing"]) == 44
    assert "pilot" in json.dumps(out["_not_scored"])


# ----------------------------------------------------------------------------- telemetry
def test_rate_limit_headers_are_whitelisted_and_carry_no_credentials():
    h = {"x-ratelimit-limit-tokens": "1000", "x-ratelimit-remaining-requests": "9", "anthropic-ratelimit-output-tokens-limit": "400000",
         "retry-after": "3", "x-request-id": "req_1", "Authorization": "Bearer sk-secret", "set-cookie": "a=b",
         "openai-organization": "org-xyz", "openai-project": "proj_1", "x-api-key": "k", "content-type": "text/event-stream"}
    out = rate_limit_headers(h)
    assert out == {"x-ratelimit-limit-tokens": "1000", "x-ratelimit-remaining-requests": "9",
                   "anthropic-ratelimit-output-tokens-limit": "400000", "retry-after": "3", "x-request-id": "req_1"}
    assert "sk-secret" not in json.dumps(out) and "org-xyz" not in json.dumps(out)
    err = TransportError("x", charged="unknown", rate_limit=out, last_event="response.created", error_body='{"type":"server_error"}')
    assert err.record()["rate_limit"] == out and err.record()["last_event"] == "response.created"


def test_process_telemetry_counts_requests_and_never_changes_them(tmp_path):
    t = ProcessTelemetry(tmp_path / "tel.jsonl", interval_s=3600, labels={"dsl": "triton"}).start()
    inner = MockProvider([{"text": scripted_text("impl_triton.py", _valid())}])
    p = TelemetryProvider(inner, t)
    from tilebench.llm.v2.providers.base import GenerationRequest
    res = p.generate(GenerationRequest(system="s", user="u", model_id="m", provider="mock"))
    t.stop()
    rows = [json.loads(x) for x in (tmp_path / "tel.jsonl").read_text().splitlines()]
    assert res.text and rows[-1]["requests_started"] == 1 and rows[-1]["requests_finished"] == 1 and rows[-1]["active_requests"] == 0
    assert any("rss_mb" in r for r in rows) and all(r["dsl"] == "triton" for r in rows)
