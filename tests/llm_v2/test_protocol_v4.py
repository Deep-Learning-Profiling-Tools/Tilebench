"""Protocol revision 4: 20 frozen cases per candidate, suite evaluator, PrecisionGuard fix,
checker false-positive fixes, Claude Code compliance adjudication, richer usage, hard
global evaluation backlog. No GPU, no network."""
from __future__ import annotations

import collections
import json
import math
import threading
import time
from pathlib import Path

import pytest
import torch

from tilebench.llm.v2 import cli
from tilebench.llm.v2.contracts.loader import load_contract
from tilebench.llm.v2.devtools import synthetic_context
from tilebench.llm.v2.evaluation import worker
from tilebench.llm.v2.evaluation.job import build_evaluation_job
from tilebench.llm.v2.evaluation.launcher import MockEvaluator, classify_no_result, mock_suite
from tilebench.llm.v2.manifests import schema as ms
from tilebench.llm.v2.metrics import cost as C
from tilebench.llm.v2.metrics import efficiency as eff
from tilebench.llm.v2.orchestration import state_machine as sm
from tilebench.llm.v2.orchestration.campaign import new_trajectory_state, protocol_identity
from tilebench.llm.v2.orchestration.runner import RunnerConfig, TrajectoryRunner
from tilebench.llm.v2.orchestration.state import AttemptRecord, TrajectoryState
from tilebench.llm.v2.prompts.renderer import render_initial, render_refinement, render_system
from tilebench.llm.v2.providers.mock import MockProvider, scripted_text
from tilebench.llm.v2.providers.usage import normalize_anthropic_messages, normalize_openai_responses
from tilebench.llm.v2.tasks import case_sets, representative
from tilebench.llm.v2.tasks.support import eligibility, task_table
from tilebench.llm.v2.validation.contract_checks import check_compliance

REV3_NON_FP16 = {"flash_decode": "fp32", "fused_activation": "fp32", "kl_divergence": "fp32", "linear_self_attention": "fp32",
                 "quantize_global": "fp32", "top_k_selection": "fp32", "dequantize_rowwise": "int8", "matmul_int8": "int8",
                 "histogramming": "int32", "radix_sort": "int32"}


# ----------------------------------------------------------------------------- protocol / task identity
def test_representative_dtypes_unchanged_and_every_task_has_exactly_20_frozen_cases(study, folds):
    sel = representative.selected()
    assert len(sel) == 45 and {op: dt for op, dt in sel.items() if dt != "fp16"} == REV3_NON_FP16
    data = case_sets.load_manifest()
    assert data["status"] == "frozen" and data["cases_per_task"] == 20
    for op, entry in data["operators"].items():
        assert entry["dtype"] == sel[op] and entry["n_cases"] == 20 and len(entry["cases"]) == 20
        assert [c["case_id"] for c in entry["cases"]] == [c["case_id"] for c in case_sets.operator_cases(op, sel[op])]
        assert len({c["case_id"] for c in entry["cases"]}) == 20                     # no duplicate, nothing synthesized


def test_case_manifest_refuses_a_changed_case_set():
    data = json.loads(json.dumps(case_sets.load_manifest()))
    data["operators"]["vector_add"]["cases"][0]["params"]["n"] += 1
    with pytest.raises(ms.ManifestError):
        case_sets.validate_manifest(data)


def test_base_and_enhanced_use_the_same_case_ids_and_one_job_covers_all_cases(study, folds):
    b = eligibility("softmax", "fp16", "B200", "triton", study, folds)
    rows = [e for e in task_table(study, folds, ["softmax"]) if e.key.device == "B200" and e.key.dsl == "triton"]
    assert len(rows) == 1 and rows[0].case_set_id == b.case_set_id and len(b.cases) == 20
    ctx_b, job_b, _ = synthetic_context("softmax", "fp16", "B200", "triton", "base", study, folds)
    ctx_e, job_e, _ = synthetic_context("softmax", "fp16", "B200", "triton", "enhanced", study, folds)
    assert job_b.case_ids() == job_e.case_ids() == [c["case_id"] for c in b.cases]
    assert ctx_b.n_cases == ctx_e.n_cases == 20


def test_prompt_states_the_domain_never_the_case_list_and_forbids_shape_tables(study, folds):
    ctx, job, _ = synthetic_context("softmax", "fp16", "B200", "triton", "base", study, folds)
    s, u = render_system(ctx), render_initial(ctx)
    assert "do not hard-code a table keyed to particular benchmark shapes" in s and "Do not time, search or autotune" in s
    assert "`n_rows` = `2048` in every case" in u and "`n_cols`: an integer from `512` to `10240`" in u
    for c in job.cases:
        assert c["case_id"] not in u and c["case_id"] not in s
    assert "per-case" not in u.lower() or "Per-case runtimes are not reported" in s


# ----------------------------------------------------------------------------- 20-case evaluator (CPU worker)
@pytest.fixture
def cpu_suite(monkeypatch, tmp_path):
    from tilebench.llm.v2.evaluation.timing import TimingRecord
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    import tilebench.data.tensors as tensors
    monkeypatch.setattr(tensors, "get_generator", lambda op: (lambda n, dtype: (torch.randn(n, dtype=dtype), torch.randn(n, dtype=dtype))))
    import tilebench.llm.v2.evaluation.timing as timing
    calls = []

    def fake_measure(f, **kw):
        f()
        calls.append(kw.get("label"))
        v = 0.1 * len(calls)
        return TimingRecord(1, 3, [v, v, v], v, True, True, True, None, 4, "graph", True, 253)
    monkeypatch.setattr(timing, "measure", fake_measure)
    rules = {"schema": "tilebench-evaluator-rules/1", "forbidden_substitutions": [], "required_evidence": [],
             "allowed_torch_calls": [], "mutation": {"inputs_mutated": []}, "outputs": {"aliasing": "none"},
             "required_stages": [], "timing_boundary": {}, "tolerance_source": "config.verify", "operator": "vector_add"}

    def run(src: str, ns=(64, 128, 256)) -> dict:
        p = tmp_path / "impl_triton.py"
        p.write_text(src)
        cases = [{"case_id": f"c{i}", "case_index": i, "params": {"n": n}, "problem_size": n} for i, n in enumerate(ns)]
        job = build_evaluation_job(operator="vector_add", dtype="fp32", cases=cases, dsl="triton", device="B200", arch=None,
                                   rules=rules, study=ms.load_study())
        return worker.run_job(job.worker_job(source_path=str(p), sandbox_dir=str(tmp_path), seed=1, round_index=1, attempt=1))
    run.calls = calls
    return run


def test_all_cases_valid_round_is_valid_with_per_case_samples_configs_and_geomean(cpu_suite):
    src = ("import torch\n_c = {}\ndef run(x, y):\n    _c['BLOCK'] = 1024 if x.numel() >= 128 else 256\n    return x + y\n"
           "def get_last_config():\n    return dict(_c)\n")
    r = cpu_suite(src)
    assert r["status"] == "valid" and r["valid_cases"] == r["cases_total"] == 3 and r["cases_evaluated"] == 3
    assert [c["config"] for c in r["cases"]] == [{"BLOCK": 256}, {"BLOCK": 1024}, {"BLOCK": 1024}]    # deterministic per-shape config
    assert r["configs_distinct"] == [{"BLOCK": 256}, {"BLOCK": 1024}]
    for c in r["cases"]:
        assert len(c["latency_ms_samples"]) == 3 and len(c["config_reads"]) == 3 and c["stages"]["timing"]["timed"] == 3
    means = [c["latency_ms_mean"] for c in r["cases"]]
    assert abs(r["latency_ms_geomean"] - math.exp(sum(map(math.log, means)) / 3)) < 1e-12
    assert cpu_suite.calls == ["case000", "case001", "case002"]                 # one timing per case, case-labelled profile


def test_one_invalid_case_makes_the_round_performance_invalid_and_keeps_the_other_cases(cpu_suite):
    src = "import torch\ndef run(x, y):\n    return x + y if x.numel() != 128 else x - y\ndef get_last_config():\n    return {}\n"
    r = cpu_suite(src)
    assert r["status"] == "numerical_error" and r["valid_cases"] == 2 and r["cases_evaluated"] == 3
    assert r["latency_ms_geomean"] is None and r["first_failing_case"]["params"] == {"n": 128}
    assert [c["status"] for c in r["cases"]] == ["valid", "numerical_error", "valid"]


def test_contract_violation_stops_the_suite_without_fabricating_cases(cpu_suite):
    r = cpu_suite("import torch\ndef run(x, y):\n    x.add_(y)\n    return x\ndef get_last_config():\n    return {}\n")
    assert r["status"] == "contract_violation" and r["suite_stopped"]["reason"] == "contract_violation"
    assert [c["status"] for c in r["cases"]] == ["contract_violation", "not_evaluated", "not_evaluated"]


def test_suite_result_applies_to_state_and_inconsistent_suites_are_refused():
    def state():
        st = TrajectoryState(schema="s", trajectory_id="t", task={}, model="m", condition="base", config_hash="h",
                             content_hashes={}, output_file="impl_triton.py")
        sm.apply_attempt(st, 1, AttemptRecord(attempt=1, kind="initial", request_hash="r", prompt_chars=1, transport_attempts=1,
                                              verdict="clear", source_path="/x"))
        return st
    job = type("J", (), {"cases": [{"case_id": f"c{i}", "case_index": i, "params": {"n": i}} for i in range(20)]})()
    good = mock_suite(job, "valid", 1.0)
    st = state()
    sm.apply_evaluation(st, 1, good)
    rec = st.rounds[0]
    assert rec.status == "valid" and rec.valid_cases == rec.cases_total == 20 and len(rec.case_results) == 20
    assert st.best_valid["latency_ms_geomean"] == pytest.approx(rec.latency_ms_geomean)
    bad = json.loads(json.dumps(good))
    bad["latency_ms_geomean"] *= 1.5
    st2 = state()
    sm.apply_evaluation(st2, 1, bad)
    assert st2.rounds[0].status == "timing_error"
    short = json.loads(json.dumps(good))
    short["cases"] = short["cases"][:19]
    st3 = state()
    sm.apply_evaluation(st3, 1, short)
    assert st3.rounds[0].status == "timing_error"


def test_worker_death_keeps_finished_cases(tmp_path):
    (tmp_path / "progress.json").write_text(json.dumps({"phase": "timing_started", "case": 1}))
    (tmp_path / "cases.jsonl").write_text(json.dumps({"case_id": "c0", "status": "valid", "latency_ms_mean": 1.0,
                                                      "latency_ms_samples": [1.0, 1.0, 1.0], "params": {"n": 1}}) + "\n")
    cases = [{"case_id": f"c{i}", "case_index": i, "params": {"n": i}} for i in range(3)]
    r = classify_no_result(tmp_path, timed_out=False, timeout_s=3600, rc=-9, stderr="Killed", job_cases=cases)
    assert r["status"] == "runtime_error" and r["worker_killed_by_signal"] == 9
    assert [c["status"] for c in r["cases"]] == ["valid", "runtime_error", "not_evaluated"] and r["valid_cases"] == 1


def test_geometric_mean_efficiency_and_speedup_metrics():
    t_emp = {"a": 1.0, "b": 4.0}
    rounds = [{"round": 1, "attempts": [{"cost": 10}], "valid": True, "cases_total": 2, "valid_cases": 2,
               "case_latency_ms": {"a": 2.0, "b": 8.0}, "case_samples_ms": {"a": [2, 2, 2], "b": [8, 8, 8]}, "latency_ms": 4.0},
              {"round": 2, "attempts": [{"cost": 10}], "valid": True, "cases_total": 2, "valid_cases": 2,
               "case_latency_ms": {"a": 1.0, "b": 2.0}, "latency_ms": math.sqrt(2.0)}]
    c = eff.curve(t_emp, rounds)
    assert c.points[0].efficiency == pytest.approx(0.5) and c.points[1].efficiency == pytest.approx(math.sqrt(1.0 * 2.0))
    assert eff.efficiency_at(c, 10) == pytest.approx(0.5) and eff.efficiency_at(c, 20) == pytest.approx(math.sqrt(2.0))
    assert eff.geomean_ratio({"a": 10.0, "b": 40.0}, {"a": 2.0, "b": 8.0}) == pytest.approx(5.0)
    rounds[0]["valid_cases"] = 1                                               # 19/20-style round is not valid
    assert eff.curve(t_emp, rounds).points[0].valid is False


def test_job_identity_is_stable_for_the_same_case_suite(study, folds):
    from tilebench.llm.v2.evaluation.fingerprint import evaluator_fingerprint
    _, j1, _ = synthetic_context("relu", "fp16", "B200", "triton", "base", study, folds)
    _, j2, _ = synthetic_context("relu", "fp16", "B200", "triton", "base", study, folds)
    f1 = evaluator_fingerprint(j1, worker_timeout_s=3600, isolation_backend="bwrap")
    f2 = evaluator_fingerprint(j2, worker_timeout_s=3600, isolation_backend="bwrap")
    assert f1["fingerprint_sha256"] == f2["fingerprint_sha256"] and len(f1["job"]["cases"]) == 20


# ----------------------------------------------------------------------------- PrecisionGuard (CPU; flags are global state)
@pytest.fixture
def restore_flags():
    g = worker.PrecisionGuard()
    yield
    worker.restore_precision(g.baseline, g.legacy_global)


def test_reference_precision_changes_are_never_blamed_on_the_candidate(restore_flags):
    g = worker.PrecisionGuard()

    def ref_matmul_int8(x):                    # matmul_int8 reference: legacy setter inside run()
        torch.backends.cuda.matmul.allow_tf32 = False
        return x

    def ref_lsa(x):                            # linear_self_attention reference: set, then restore the old legacy value
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return x
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old
    cand = g.guard_candidate(lambda x: x)
    for ref in (ref_matmul_int8, ref_lsa):
        for _ in range(3):
            g.guard_reference(ref)(1)
            assert cand(1) == 1                # no PrecisionTampered: the guard restores the canonical family first
    assert worker.precision_diff(g.baseline, worker.precision_state()) == {}


@pytest.mark.parametrize("tamper", ["legacy_tf32", "new_api", "float32_matmul_precision", "cudnn_tf32", "default_dtype",
                                    "deterministic"])
def test_candidates_that_really_change_precision_state_are_still_violations(restore_flags, tamper):
    g = worker.PrecisionGuard()

    def cand(x):
        if tamper == "legacy_tf32":
            torch.backends.cuda.matmul.allow_tf32 = not torch.backends.cuda.matmul.allow_tf32
        elif tamper == "new_api":
            torch.backends.cuda.matmul.fp32_precision = "tf32" if g.baseline.get("cuda.matmul.fp32_precision") != "tf32" else "ieee"
        elif tamper == "float32_matmul_precision":
            torch.set_float32_matmul_precision("medium")
        elif tamper == "cudnn_tf32":
            torch.backends.cudnn.allow_tf32 = not torch.backends.cudnn.allow_tf32
        elif tamper == "default_dtype":
            torch.set_default_dtype(torch.float64)
        else:
            torch.use_deterministic_algorithms(True)
        return x
    if tamper == "new_api" and worker.precision_api_family() != "fp32_precision":
        pytest.skip("installed torch has no fp32_precision API")
    with pytest.raises(worker.PrecisionTampered):
        g.guard_candidate(cand)(1)


# ----------------------------------------------------------------------------- checker false positives (generic fixes)
def _verdict(src: str, dsl: str, op: str):
    return check_compliance(src, dsl, load_contract(op).rules)


def test_decorator_after_a_subscript_is_not_an_infix_matmul():
    src = ("import torch\nimport cuda.tile as ct\nConstInt = ct.Constant[int]\n\n\n@ct.kernel\ndef k(a, b, c, BM: ConstInt):\n"
           "    pass\n\ndef run(a, b):\n    c = torch.empty((4, 4), device=a.device)\n    return c\n\n"
           "def get_last_config():\n    return {}\n")
    r = _verdict(src, "cutile", "matmul_int8")
    assert r.verdict != "review_required" and any("not an infix matrix-multiply" in f or "spans a line break" in f for f in r.audit_flags())


def test_host_matmul_on_tensors_is_still_confirmed():
    src = "import torch\ndef run(a, b):\n    return a @ b\ndef get_last_config():\n    return {}\n"
    assert _verdict(src, "triton", "matmul_int8").verdict == "confirmed_violation"


def test_a_forbidden_name_used_only_as_the_candidates_own_identifier_is_not_delegation():
    src = ("import torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef _swiglu_kernel(x, o, n):\n    pass\n\n"
           "def _launch_swiglu(x, o):\n    _swiglu_kernel[(1,)](x, o, x.numel())\n\ndef run(x):\n    o = torch.empty_like(x)\n"
           "    _launch_swiglu(x, o)\n    return o\n\ndef get_last_config():\n    return {}\n")
    assert _verdict(src, "triton", "swiglu").verdict != "review_required"
    lib = "import torch\nfrom liger_kernel.ops.swiglu import LigerSiLUMulFunction\ndef run(x):\n    return x\ndef get_last_config():\n    return {}\n"
    assert _verdict(lib, "triton", "swiglu").verdict in ("review_required", "confirmed_violation")


def test_lock_inside_a_keyword_or_local_name_is_not_a_fixup_semaphore():
    src = ("import torch\nfrom triton.tools.tensor_descriptor import TensorDescriptor\ndef run(a, b):\n"
           "    d = TensorDescriptor(base=a, shape=[4, 4], strides=[4, 1], block_shape=[2, 2])\n"
           "    locks = torch.zeros(4, dtype=torch.int32, device=a.device)\n    return torch.empty((4, 4), device=a.device)\n"
           "def get_last_config():\n    return {}\n")
    r = _verdict(src, "triton", "streamk_matmul")
    assert not any("Semaphores" in i for i in r.review_items())


def test_guarded_contiguous_is_audit_only_and_unguarded_stays_review():
    guarded = ("import torch\ndef run(X, eps):\n    if not X.is_contiguous():\n        X = X.contiguous()\n"
               "    return torch.empty_like(X)\ndef get_last_config():\n    return {}\n")
    plain = "import torch\ndef run(X, eps):\n    X = X.contiguous()\n    return torch.empty_like(X)\ndef get_last_config():\n    return {}\n"
    assert not any("contiguous" in i for i in _verdict(guarded, "triton", "l2_norm").review_items())
    assert any("contiguous" in i for i in _verdict(plain, "triton", "l2_norm").review_items())


def test_compiled_kernel_cache_is_not_a_tensor_cache_but_an_output_cache_still_is():
    kernel_cache = ("import torch\nimport tilelang\nimport tilelang.language as T\n\n@tilelang.jit\ndef _build(R, C):\n"
                    "    @T.prim_func\n    def k(A: T.Tensor((R, C), 'float16')):\n        pass\n    return k\n\n_kernels = {}\n\n"
                    "def _get(R, C):\n    key = (R, C)\n    k = _kernels.get(key)\n    if k is None:\n        k = _build(R, C)\n"
                    "        _kernels[key] = k\n    return k\n\ndef run(x, rows, cols):\n    k = _get(int(rows), int(cols))\n"
                    "    return torch.empty_like(x)\n\ndef get_last_config():\n    return {}\n")
    r = _verdict(kernel_cache, "tilelang", "gaussian_blur")
    assert not any("holds tensor data" in i for i in r.review_items()) and any("caches a compiled" in f for f in r.audit_flags())
    out_cache = ("import torch\n_cache = {}\ndef run(x, rows, cols):\n    k = x.data_ptr()\n    if k in _cache:\n        return _cache[k]\n"
                 "    out = torch.empty_like(x)\n    _cache[k] = out\n    return out\ndef get_last_config():\n    return {}\n")
    assert any("holds tensor data" in i for i in _verdict(out_cache, "tilelang", "gaussian_blur").review_items())


POOL_CONFIG = ("import torch\nimport triton\nimport triton.language as tl\n_last_config = {}\n\n@triton.jit\n"
               "def _k(X, Y, n, BLOCK: tl.constexpr):\n    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
               "    tl.store(Y + o, tl.load(X + o, mask=o < n), mask=o < n)\n\n"
               "def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):\n    global _last_config\n"
               "    oh = (H + 2 * padding - kernel_size) // stride + 1\n    ow = (W + 2 * padding - kernel_size) // stride + 1\n"
               "    total = N * C * oh * ow\n    out = torch.empty(total, dtype=input.dtype, device=input.device)\n"
               "    grid = (triton.cdiv(total, 256),)\n    _last_config = {'BLOCK': 256, 'grid': grid}\n"
               "    _k[grid](input, out, total, BLOCK=256)\n    return out\n\ndef get_last_config():\n    return dict(_last_config)\n")


def test_scalar_run_inputs_are_configuration_but_tensor_and_output_caches_stay_review():
    from tilebench.llm.v2.tasks.input_kinds import scalar_positions
    rules = load_contract("2d_max_pooling").rules
    pos = scalar_positions("2d_max_pooling")
    assert pos == frozenset(range(1, 8))                       # input, N, C, H, W, kernel_size, stride, padding
    # a launch grid computed from the scalar inputs, kept for get_last_config(): configuration (checker v3 flagged it)
    assert any("holds tensor data" in i for i in check_compliance(POOL_CONFIG, "triton", rules).review_items())
    assert not any("holds tensor data" in i for i in check_compliance(POOL_CONFIG, "triton", rules, scalar_positions=pos).review_items())
    # the tensor input or the output in persistent state stays a possible result cache
    keeps_input = POOL_CONFIG.replace("_last_config = {'BLOCK': 256, 'grid': grid}", "_last_config = {'BLOCK': 256, 'x': input}")
    keyed_out = POOL_CONFIG.replace("    return out\n\ndef get", "    _last_config[(N, C, H, W)] = out\n    return out\n\ndef get")
    for src in (keeps_input, keyed_out):
        assert any("holds tensor data" in i for i in check_compliance(src, "triton", rules, scalar_positions=pos).review_items())
    # only the positions the task passes as numbers: a scalar position never covers a tensor input
    assert 0 not in pos and scalar_positions("vector_add") == frozenset()


def test_input_kinds_manifest_matches_the_case_sets_and_the_generators():
    from tilebench.llm.v2.tasks import input_kinds
    m = input_kinds.load_manifest()
    cs = ms.load_case_sets()["operators"]
    assert set(m["operators"]) == set(cs) and len(cs) == 45
    assert all(e["kinds"][0] == "tensor" for e in m["operators"].values())
    if not torch.cuda.is_available():
        pytest.skip("generators need the GPU")
    for op, e in sorted(m["operators"].items()):                 # first case of every task, as the evaluator builds it
        assert input_kinds.derive(op, {**cs[op], "cases": cs[op]["cases"][:1]}) == e["kinds"], op


# ----------------------------------------------------------------------------- Claude Code adjudication
def _formal_review_state(study, folds, tmp_path):
    ctx, job, _ = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    job.rules = load_contract("vector_add").rules
    el = eligibility("vector_add", "fp16", "B200", "triton", study, folds)
    st = new_trajectory_state(el, ctx, "gpt", "base", "h", load_contract("vector_add").sha256, "t", run_type="formal",
                              campaign="unit_rev4", protocol=protocol_identity(study))
    review_src = ("# MOCK: valid 1.0\nimport torch\ndef run(x, y):\n    return torch.ops.aten.add(x, y)\n"
                  "def get_last_config():\n    return {}\n")
    prov = MockProvider([{"text": scripted_text("impl_triton.py", review_src)},
                         {"text": scripted_text("impl_triton.py", "# MOCK: valid 1.0\ndef run(*a): pass\ndef get_last_config(): return {}")}])
    cfg = RunnerConfig(model_id="m", provider_name="mock", settings={}, feedback_limits=study["feedback"], retry_backoff_s=0.0)
    tdir = tmp_path / "traj"
    r = TrajectoryRunner(state=st, tdir=tdir, ctx=ctx, provider=prov, evaluator=MockEvaluator(), job=job, cfg=cfg)
    r.step()
    assert st.status == "review_required"
    return r, st, prov, tdir


def test_claude_code_is_the_formal_adjudicator_and_compliant_evaluates_the_archived_candidate(study, folds, tmp_path):
    r, st, prov, tdir = _formal_review_state(study, folds, tmp_path)
    packet = cli.review_evidence(tdir, st)
    assert packet["review_items"] and not any(k in json.dumps(packet).lower() for k in ('"latency', '"speedup', '"usd', '"t_emp'))
    assert cli.main(["review-resolve", "--trajectory-dir", str(tdir), "--decision", "compliant", "--reviewer", "a-human",
                     "--note", "x"]) == 2                                        # formal: no human adjudicator
    assert cli.main(["review-resolve", "--trajectory-dir", str(tdir), "--decision", "compliant"]) == 2   # rationale required
    assert cli.main(["review-resolve", "--trajectory-dir", str(tdir), "--decision", "compliant",
                     "--note", "torch.ops.aten.add is the candidate's elementwise add helper; permitted by the contract"]) == 0
    st2 = TrajectoryState.load(tdir / "trajectory.json")
    rec = st2.rounds[0]
    assert rec.reviews[-1]["reviewer"] == "claude-code" and rec.reviews[-1]["reviewer_api_calls"] == 0
    assert rec.reviews[-1]["candidate_sha256"] and rec.reviews[-1]["contract_sha256"] and rec.reviews[-1]["rules_sha256"]
    r.state = st2
    n_requests = len(prov.requests)
    r.step()                                                                     # evaluate the archived candidate
    assert st2.rounds[0].status == "valid" and len(prov.requests) == n_requests  # no new generation


def test_violation_closes_the_round_without_repair_and_the_next_round_sees_it(study, folds, tmp_path):
    r, st, prov, tdir = _formal_review_state(study, folds, tmp_path)
    assert cli.main(["review-resolve", "--trajectory-dir", str(tdir), "--decision", "violation",
                     "--note", "the operator computation is delegated to an aten operator"]) == 0
    st2 = TrajectoryState.load(tdir / "trajectory.json")
    assert st2.rounds[0].status == "contract_violation" and len(st2.rounds[0].attempts) == 1
    r.state = st2
    r.step()
    req = prov.requests[-1]
    assert len(prov.requests) == 2 and "delegated to an aten operator" in req.user and "Optimization round 2" in req.user


# ----------------------------------------------------------------------------- usage / cost
def test_openai_richer_usage_is_normalized_without_double_counting():
    raw = {"input_tokens": 14764, "input_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 13761},
           "output_tokens": 5663, "output_tokens_details": {"reasoning_tokens": 5038}, "total_tokens": 20427}
    u = normalize_openai_responses(raw, {"id": "resp_1", "model": "gpt-6.1-sol", "service_tier": "default",
                                         "prompt_cache_retention": "24h"})
    assert (u.logical_input_total, u.cached_input, u.cache_write_input, u.uncached_input) == (14764, 1000, 13761, 3)
    assert (u.logical_output_total, u.reasoning_or_thinking_output, u.non_reasoning_output) == (5663, 5038, 625)
    assert u.logical_total == 20427 == u.total_tokens_if_reported and u.service_tier == "default"
    assert u.cache_diagnostics == {"prompt_cache_retention": "24h"} and u.response_id == "resp_1"
    usd = C.request_cost("openai", "gpt-6.1-sol", raw)
    assert usd["tokens"]["output"] == 5663 and usd["usd_cost_status"] == "exact_from_usage_and_frozen_price"
    expect = (3 * 2.0 + 1000 * 0.10 + 13761 * 2.50 + 5663 * 10.0) / 1e6               # reasoning is inside output, never added
    assert usd["estimated_cost_usd"] == pytest.approx(expect)


def test_anthropic_thinking_and_server_tools_are_read_when_returned_and_null_otherwise():
    raw = {"input_tokens": 100, "cache_read_input_tokens": 50, "cache_creation_input_tokens": 30,
           "cache_creation": {"ephemeral_5m_input_tokens": 20, "ephemeral_1h_input_tokens": 10}, "output_tokens": 400,
           "output_tokens_details": {"thinking_tokens": 300}, "server_tool_use": {"web_search_requests": 0},
           "service_tier": "standard", "inference_geo": "global"}
    u = normalize_anthropic_messages(raw, {"id": "msg_1", "model": "claude-opus-5-5", "diagnostics": {"cache_miss_reason": "x"}})
    assert (u.logical_input_total, u.uncached_input, u.cached_input, u.cache_write_input) == (180, 100, 50, 30)
    assert (u.logical_output_total, u.reasoning_or_thinking_output, u.non_reasoning_output) == (400, 300, 100)
    assert u.logical_total == 580 and u.server_tool_counts == {"web_search_requests": 0} and u.inference_geo == "global"
    assert u.cache_diagnostics["diagnostics"] == {"cache_miss_reason": "x"}
    usd = C.request_cost("anthropic", "claude-opus-5-5", raw)
    assert usd["tokens"]["output"] == 400                                         # thinking never added again
    bare = {k: v for k, v in raw.items() if k not in ("output_tokens_details", "server_tool_use")}
    b = normalize_anthropic_messages(bare)
    assert b.reasoning_or_thinking_output is None and b.non_reasoning_output is None and b.server_tool_counts is None


# ----------------------------------------------------------------------------- pipeline (scheduler, mock provider/evaluator)
def test_generation_overlaps_other_evaluations_same_trajectory_waits_and_backlog_is_hard_bounded(study, folds, tmp_path, monkeypatch):
    from tilebench.llm.v2.orchestration import campaign
    from tilebench.llm.v2.orchestration.campaign import CampaignSpec, Preflight
    from tilebench.llm.v2.orchestration.scheduler import run_scheduled
    monkeypatch.setattr(campaign, "preflight", lambda *a, **k: Preflight(device="B200", dsl=k.get("dsl", "triton"), condition="base",
                                                                         ok=True, run_type="validation"))
    lock = threading.Lock()
    log = collections.defaultdict(list)

    class Prov:
        name, usage_schema = "mock", "openai_responses_v1"

        def generate(self, req):
            t0 = time.time()
            time.sleep(0.05)
            out = req.user.split("output file: `")[1].split("`")[0] if "output file: `" in req.user else \
                req.user.split("output file `")[1].split("`")[0]
            res = MockProvider([{"text": scripted_text(out, "# MOCK: valid 1.0\ndef run(*a): pass\ndef get_last_config(): return {}")}]).generate(req)
            with lock:
                log[req.metadata["trajectory"]].append(("gen", req.metadata["round"], t0, time.time()))
            return res

    class Ev(MockEvaluator):
        report, timeout_s = {"backend": "bwrap"}, 3600
        dev = threading.Lock()

        def evaluate(self, source_path, job, round_index, attempt, *, archive_dir=None):
            with self.dev:                                                     # the device lock
                t0 = time.time()
                time.sleep(0.03)
                out = super().evaluate(source_path, job, round_index, attempt, archive_dir=archive_dir)
            with lock:
                log[job.identity["trajectory_id"]].append(("eval", round_index, t0, time.time()))
            return out
    spec = CampaignSpec(name="unit_sched_v4", run_type="validation", device="B200", dsl="triton", condition="base", model="gpt",
                        operators=["vector_add", "relu"], out_root=tmp_path, resume=True, isolation="none")
    out = run_scheduled(spec, ["triton", "cutile"], models=["gpt", "claude"], initial=3, max_limit=8, max_pending=3,
                        log=lambda s: None, providers={"gpt": Prov(), "claude": Prov()}, evaluator=Ev(), telemetry_interval_s=0.2)
    assert not out["errors"] and not out["refused_tracks"] and set(out["statuses"].values()) == {"complete"}
    assert len(out["statuses"]) == 8 and out["eval_budget"]["max_held"] <= 3 and out["max_pending_evaluations_seen"] <= 3
    for tid, evs in log.items():
        evs.sort(key=lambda e: e[2])
        for r in range(1, 5):
            ev_end = next(e[3] for e in evs if e[0] == "eval" and e[1] == r)
            gen_next = next(e[2] for e in evs if e[0] == "gen" and e[1] == r + 1)
            assert gen_next >= ev_end                                           # a trajectory waits for its own evaluation
    gens = [e for evs in log.values() for e in evs if e[0] == "gen"]
    evals = [e for evs in log.values() for e in evs if e[0] == "eval"]
    assert any(g[2] < v[3] and v[2] < g[3] for g in gens for v in evals)       # other trajectories generate meanwhile


def test_concurrency_ramp_is_gated_on_the_evaluation_backlog_not_on_budget_reservations(study, folds, tmp_path, monkeypatch):
    """With more trajectories than budget slots, every slot is reserved by a trajectory waiting to send; the limiter
    must still ramp while few candidates wait for the device (the B200 formal launch stayed at 3 otherwise)."""
    from tilebench.llm.v2.orchestration import campaign, scheduler
    from tilebench.llm.v2.orchestration.campaign import CampaignSpec, Preflight
    monkeypatch.setattr(campaign, "preflight", lambda *a, **k: Preflight(device="B200", dsl=k.get("dsl", "triton"), condition="base",
                                                                         ok=True, run_type="validation"))
    seen = {}
    real_limiter, real_tracker = scheduler.AdaptiveLimiter, scheduler.EvalTracker

    class Lim(real_limiter):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen.setdefault("pending_fns", []).append(k["pending_fn"])

    class Tracker(real_tracker):
        def __init__(self, inner):
            super().__init__(inner)
            seen["tracker"] = self

    class Budget(scheduler.EvalBudget):
        def __init__(self, cap):
            super().__init__(cap)
            seen["budget"] = self
    monkeypatch.setattr(scheduler, "AdaptiveLimiter", Lim)
    monkeypatch.setattr(scheduler, "EvalTracker", Tracker)
    monkeypatch.setattr(scheduler, "EvalBudget", Budget)
    spec = CampaignSpec(name="unit_ramp_gate", run_type="validation", device="B200", dsl="triton", condition="base", model="gpt",
                        operators=["vector_add"], out_root=tmp_path, resume=True, isolation="none")
    good = scripted_text("impl_triton.py", "# MOCK: valid 1.0\ndef run(*a): pass\ndef get_last_config(): return {}")

    class Ev(MockEvaluator):
        report, timeout_s = {"backend": "bwrap"}, 3600
    scheduler.run_scheduled(spec, ["triton"], models=["gpt"], max_pending=4, log=lambda s: None,
                            providers={"gpt": MockProvider([{"text": good}] * 5)}, evaluator=Ev(), telemetry_interval_s=0.2)
    fn, tracker, budget = seen["pending_fns"][0], seen["tracker"], seen["budget"]
    budget.used, tracker.pending = 4, 1                     # every slot reserved, one candidate at the device
    assert fn() == 1 < 4 / 2
    tracker.pending = 3
    assert fn() == 3


def test_scheduler_continues_a_trajectory_after_the_adjudicator_resolves_its_review(study, folds, tmp_path, monkeypatch):
    from tilebench.llm.v2.orchestration import campaign
    from tilebench.llm.v2.orchestration.campaign import CampaignSpec, Preflight
    from tilebench.llm.v2.orchestration.scheduler import run_scheduled
    monkeypatch.setattr(campaign, "preflight", lambda *a, **k: Preflight(device="B200", dsl="triton", condition="base", ok=True,
                                                                         run_type="validation"))
    review = "# MOCK: valid 1.0\nimport torch\ndef run(x, y):\n    return torch.ops.aten.add(x, y)\ndef get_last_config():\n    return {}\n"
    good = "# MOCK: valid 1.0\ndef run(*a): pass\ndef get_last_config(): return {}"

    class Prov:
        name, usage_schema = "mock", "openai_responses_v1"

        def generate(self, req):
            text = review if req.metadata["round"] == 1 else good
            return MockProvider([{"text": scripted_text("impl_triton.py", text)}]).generate(req)

    class Ev(MockEvaluator):
        report, timeout_s = {"backend": "bwrap"}, 3600
    spec = CampaignSpec(name="unit_review_resume", run_type="validation", device="B200", dsl="triton", condition="base", model="gpt",
                        operators=["vector_add"], out_root=tmp_path, resume=True, isolation="none")
    result = {}
    th = threading.Thread(target=lambda: result.update(run_scheduled(spec, ["triton"], models=["gpt"], max_pending=2, log=lambda s: None,
                                                                     providers={"gpt": Prov()}, evaluator=Ev(),
                                                                     telemetry_interval_s=0.2, review_poll_s=0.2)))
    th.start()
    tpath = None
    for _ in range(200):
        hits = list((tmp_path / "unit_review_resume").glob("base/B200/triton/gpt/vector_add/*/*/trajectory.json"))
        if hits and json.loads(hits[0].read_text())["status"] == "review_required":
            tpath = hits[0]
            break
        time.sleep(0.1)
    assert tpath is not None
    assert cli.main(["review-resolve", "--trajectory-dir", str(tpath.parent), "--decision", "compliant",
                     "--note", "aten add helper is an elementwise add on the inputs; not a library operator substitution"]) == 0
    th.join(timeout=60)
    assert not th.is_alive() and result["resumed_after_review"] and set(result["statuses"].values()) == {"complete"}
    st = json.loads(tpath.read_text())
    assert st["status"] == "complete" and st["rounds"][0]["status"] == "valid" and st["rounds"][0]["reviews"][0]["reviewer"] == "claude-code"
