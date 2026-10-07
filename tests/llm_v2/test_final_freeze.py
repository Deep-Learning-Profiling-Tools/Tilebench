"""Blocking fixes of FINAL_FREEZE_AND_START_B200.md §1 and the frozen
execution policies of §5: frozen folds for formal Base, BLAS-stack gate,
process-wide precision protection, scoring-only calibration, formal
capture-failure policy and the frozen wall-clock limit."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from tilebench.llm.v2.evaluation import job as evjob
from tilebench.llm.v2.evaluation import worker
from tilebench.llm.v2.manifests import schema as ms
from tilebench.llm.v2.orchestration import campaign
from tilebench.llm.v2.prompts import feedback
from tilebench.llm.v2.validation import static_checks
from tilebench.paths import REPO_ROOT

SNAPSHOTS = REPO_ROOT / "docs" / "llm_v2" / "prompt_snapshots"


# ----------------------------------------------------------------------------- §1.1 frozen folds for formal Base
def test_formal_base_refuses_proposed_folds_and_accepts_frozen(study, monkeypatch):
    real = ms.load_folds()
    proposed = {**real, "status": "proposed", "approved_by": None}
    monkeypatch.setattr(ms, "load_folds", lambda: proposed)
    pf = campaign.preflight("B200", "triton", "base", live=True, study=study, run_type="formal", operators=["vector_add"])
    assert any("fold assignment is not frozen" in b and "formal Base" in b for b in pf.blockers)
    pfv = campaign.preflight("B200", "triton", "base", live=True, study=study, run_type="validation",
                             models_selected=("gpt",), provider="openai", operators=["vector_add"])
    assert not any("fold" in b for b in pfv.blockers) and any("unfrozen folds" in w for w in pfv.warnings)
    frozen_no_owner = {**real, "status": "frozen", "approved_by": None}
    monkeypatch.setattr(ms, "load_folds", lambda: frozen_no_owner)
    pf2 = campaign.preflight("B200", "triton", "base", live=True, study=study, run_type="formal", operators=["vector_add"])
    assert any("frozen without approved_by" in b for b in pf2.blockers)
    monkeypatch.setattr(ms, "load_folds", lambda: real)
    assert real["status"] == "frozen" and real["approved_by"]
    pf3 = campaign.preflight("B200", "triton", "base", live=True, study=study, run_type="formal", operators=["vector_add"])
    assert not any("fold" in b for b in pf3.blockers)


# ----------------------------------------------------------------------------- §1.2 BLAS stack gate
def test_formal_preflight_fails_closed_on_a_mixed_blas_stack(study, monkeypatch):
    from tilebench.llm.v2.calibration import environment
    bad = {"ok": False, "error": "BLAS libraries loaded from different directories: ['/wheel', '/usr/local/cuda-13.2']",
           "libraries": ["/wheel/libcublas.so.13", "/usr/local/cuda-13.2/.../libcublasLt.so.13.4.0.1"],
           "blas_dirs": ["/wheel", "/usr/local/cuda-13.2"], "checks": {"int8 _int_mm": "CUBLAS_STATUS_NOT_INITIALIZED"},
           "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"}
    monkeypatch.setattr(environment, "blas_stack_check", lambda env=None, timeout=300.0: bad)
    pf = campaign.preflight("B200", "triton", "base", live=True, study=study, run_type="formal", operators=["vector_add"])
    assert any("BLAS stack" in b and "different directories" in b for b in pf.blockers)
    assert pf.facts["blas_stack"]["blas_dirs"] == ["/wheel", "/usr/local/cuda-13.2"]
    pfv = campaign.preflight("B200", "triton", "base", live=True, study=study, run_type="validation",
                             models_selected=("gpt",), provider="openai", operators=["vector_add"])
    assert not any("BLAS" in b for b in pfv.blockers) and any("BLAS stack" in w for w in pfv.warnings)
    good = {"ok": True, "libraries": ["/wheel/libcublas.so.13", "/wheel/libcublasLt.so.13"], "blas_dirs": ["/wheel"],
            "checks": {"int8 _int_mm": "ok"}, "LD_LIBRARY_PATH": "<unset>"}
    monkeypatch.setattr(environment, "blas_stack_check", lambda env=None, timeout=300.0: good)
    pf2 = campaign.preflight("B200", "triton", "base", live=True, study=study, run_type="formal", operators=["vector_add"])
    assert not any("BLAS" in b for b in pf2.blockers) and pf2.facts["blas_stack"]["ok"] is True


# ----------------------------------------------------------------------------- §1.3 process-wide precision state
def test_precision_state_reads_are_repeatable():
    a = worker.precision_state()
    b = worker.precision_state()
    assert worker.precision_diff(a, b) == {} and not any(str(v).startswith("<unreadable") for v in a.values())
    if worker.precision_api_family() == "fp32_precision":     # torch >= 2.9: the per-backend fp32_precision family is canonical
        assert {"cuda.matmul.fp32_precision", "cudnn.conv.fp32_precision", "backends.fp32_precision", "cudnn.benchmark",
                "deterministic_algorithms"} <= set(a) and "cuda.matmul.allow_tf32" not in a
    else:
        assert {"cuda.matmul.allow_tf32", "cudnn.benchmark", "float32_matmul_precision", "deterministic_algorithms"} <= set(a)


def test_precision_guard_detects_candidate_changes_and_restores_before_reference():
    g = worker.PrecisionGuard()
    base = dict(g.baseline)
    calls = []

    def candidate(x):
        torch.backends.cudnn.benchmark = not base["cudnn.benchmark"]      # candidate flips evaluator state
        calls.append("cand")
        return x

    def reference(x):
        calls.append(("ref", torch.backends.cudnn.benchmark))
        return x
    try:
        with pytest.raises(worker.PrecisionTampered, match="cudnn.benchmark"):
            g.guard_candidate(candidate)(1)
        # the flag is still flipped; the guarded reference restores it before computing
        assert torch.backends.cudnn.benchmark != base["cudnn.benchmark"]
        g.guard_reference(reference)(1)
        assert calls[-1] == ("ref", base["cudnn.benchmark"]) and g.restores and g.restores[-1]["where"] == "before reference call"
        # a well-behaved candidate passes; the baseline is restored before it runs
        torch.backends.cudnn.benchmark = not base["cudnn.benchmark"]
        assert g.guard_candidate(lambda x: x)(2) == 2 and torch.backends.cudnn.benchmark == base["cudnn.benchmark"]
        rec = g.record()
        assert rec["baseline"] == base and rec["verifications"] >= 2
    finally:
        worker.restore_precision(base)


def test_static_checks_flag_precision_state_changes_as_confirmed():
    src = ("import torch\nimport torch as T\n"
           "torch.backends.cuda.matmul.allow_tf32 = True\n"
           "def run(x):\n    T.backends.cudnn.benchmark = True\n    torch.set_float32_matmul_precision('high')\n    return x\n"
           "def get_last_config():\n    return {}\n")
    rep = static_checks.analyze(src, "triton")
    conf = [e for e in rep.confirmed if e.category == "tampering"]
    assert len(conf) == 3 and {e.line for e in conf} == {3, 5, 6}
    clean = static_checks.analyze("import torch\ndef run(x):\n    return torch.empty_like(x)\ndef get_last_config():\n    return {}\n", "triton")
    assert not [e for e in clean.confirmed if e.category == "tampering"]


def test_contract_rules_forbid_precision_flag_changes_for_the_tf32_gemm():
    rules = json.loads((REPO_ROOT / "tilebench/llm/v2/contracts/data/matmul_fp32_fp16_fp8/evaluator_rules.json").read_text())
    pats = [r for r in rules["forbidden_substitutions"] if "allow_tf32" in r["pattern"] or "set_float32_matmul_precision" in r["pattern"]]
    assert pats and all(r["level"] == "confirmed" for r in pats)


# ----------------------------------------------------------------------------- §1.4 calibration is scoring-only
def test_prompts_and_feedback_carry_no_calibration_values(study):
    profile = json.loads((REPO_ROOT / "tilebench/llm/v2/manifests/calibration.yaml").read_text().split("B200:")[0] and "{}")  # noqa: F841
    forbidden = re.compile(r"T_emp|P_emp|BW_emp|calibration_id|ceiling_basis|6840\.4|empirical-roofline|B200-20261006T", re.I)
    for f in SNAPSHOTS.rglob("*.md"):
        assert not forbidden.search(f.read_text()), f
    assert {"t_emp", "p_emp", "bw_emp", "calibration_id"} <= set(study["feedback"]["forbidden_fields"])
    assert {"t_emp", "p_emp", "bw_emp", "calibration"} <= set(feedback.FORBIDDEN_FEEDBACK_KEYS)
    assert feedback._SCRUB.search("T_emp 0.5") and feedback._SCRUB.search("calibration profile")


# ----------------------------------------------------------------------------- §5 formal capture-failure policy and timeout
def test_formal_runs_turn_capture_failures_into_timing_error(study):
    assert evjob.timing_settings(study, run_type="formal")["capture_failure_policy"] == "timing_error"
    assert evjob.timing_settings(study, run_type="validation")["capture_failure_policy"] == "time_eagerly_and_flag"
    assert worker.capture_failure_status("timing_error", "graph", False) == "timing_error"
    assert worker.capture_failure_status("timing_error", "graph", True) is None
    assert worker.capture_failure_status("timing_error", "eager", False) is None            # ROCm: no capture expected
    assert worker.capture_failure_status("time_eagerly_and_flag", "graph", False) is None
    j = evjob.build_evaluation_job(operator="vector_add", dtype="fp16", params={"n": 1024}, dsl="triton", device="B200",
                                   arch="blackwell", rules={}, study=study, identity={"run_type": "formal"})
    assert j.timing["capture_failure_policy"] == "timing_error"
    jv = evjob.build_evaluation_job(operator="vector_add", dtype="fp16", params={"n": 1024}, dsl="triton", device="B200",
                                    arch="blackwell", rules={}, study=study, identity={"run_type": "validation"})
    assert jv.timing["capture_failure_policy"] == "time_eagerly_and_flag"


def test_revision_numbering_never_reuses_a_number(tmp_path):
    from tilebench.llm.v2.orchestration.runner import next_revision_name
    adir = tmp_path / "attempt_1"
    adir.mkdir()
    assert next_revision_name(adir) == "eval_0001"
    (adir / "evaluation.json").write_text("{}")                      # legacy revision 1
    assert next_revision_name(adir) == "eval_0002"
    (adir / "eval_0002").mkdir()                                     # an appended revision exists
    assert next_revision_name(adir) == "eval_0003"                   # was eval_0002 again before the fix
    (adir / "eval_0007").mkdir()
    assert next_revision_name(adir) == "eval_0008"


def test_frozen_wall_clock_limit_and_distiller_and_folds_are_recorded(study):
    assert study["evaluation"]["worker_timeout_s"] == 3600          # revision 4: one worker evaluates all 20 cases
    models = ms.load_models()
    d = models["roles"]["distiller"]
    assert (d["model_id"], d["reasoning_effort"], d["max_output_tokens"], d["status"]) == ("gpt-6.1-sol", "xhigh", 128000, "approved")
    assert ms.blockers_models(models, roles=("distiller",)) == []
    folds = ms.load_folds()
    assert folds["status"] == "frozen" and folds["approved_by"] and ms.blockers_folds(folds) == []
    assert all(len(v["operators"]) == 15 for v in folds["folds"].values())
    assert study["device_snapshots"]["B200"] == "2026-10-06"
