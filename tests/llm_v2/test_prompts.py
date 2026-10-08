"""Renderer and feedback whitelist."""
import pytest

from tilebench.llm.v2.devtools import synthetic_context
from tilebench.llm.v2.prompts import feedback as fb
from tilebench.llm.v2.prompts.renderer import TEMPLATE_DIR, RenderError, render, render_initial, render_refinement, render_system

SCRUB_TERMS = ("roofline", "T_SOL", "speedup", "stop_score", "pct_peak", "human", "torch_ms")


def test_missing_placeholder_is_an_error():
    with pytest.raises(RenderError):
        render("a {{x}} b", {})


def test_system_prompt_states_fixed_rules(study, folds):
    ctx, _, _ = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    s = render_system(ctx)
    for phrase in ("Preserve the canonical algorithm contract", "Do not invoke autotuners", "one deterministic implementation",
                   "Return only the requested implementation file", 'title="impl_triton.py"'):
        assert phrase in s


def test_initial_prompt_has_task_components_and_no_scores(study, folds):
    ctx, _, _ = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    u = render_initial(ctx)
    assert "# API reference: triton 3.6.0" in u and "# Device context: B200" in u and "canonical algorithm contract" in u
    assert "an integer from `1048576` to `20971520`, always a multiple of `1048576`" in u and "atol=" in u and "def run(" in u
    assert "20 configured cases" in u and "case_id" not in u                     # the domain, never the case list
    assert "Optimization guidance" not in u
    fb.assert_feedback_clean(u.split("# Task")[1])
    assert "flops_expr" not in u and "bytes_expr" not in u


def test_enhanced_prompt_adds_only_optimization_block(study, folds):
    base, _, _ = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    enh, _, _ = synthetic_context("vector_add", "fp16", "B200", "triton", "enhanced", study, folds)
    ub, ue = render_initial(base), render_initial(enh)
    assert "Optimization guidance" in ue and "Optimization guidance" not in ub
    assert ue.replace(ue.split("# Optimization guidance")[1].split("# Canonical algorithm contract")[0], "\n") .replace("# Optimization guidance\n", "") == ub or len(ue) > len(ub)


def test_refinement_feedback_is_runtime_only_and_scrubbed(study, folds):
    ctx, _, _ = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    prev = {"round": 2, "status": "numerical_error", "source": "def run(x): return x", "configs_distinct": [{"BLOCK": 64}],
            "latency_ms_geomean": None, "valid_cases": 7, "cases_total": 20, "cases_evaluated": 20,
            "first_failing_case": {"params": {"n": 3145728}, "status": "numerical_error"},
            "diagnostic": "mismatch at 3 elements\nroofline 55% speedup 2x T_SOL=1.0\nsecond line"}
    best = {"round": 1, "latency_ms_geomean": 1.5, "source": "def run(x): return x+0", "configs_distinct": [{"BLOCK": 32}]}
    history = [{"round": 1, "status": "valid", "latency_ms_geomean": 1.5, "valid_cases": 20, "cases_total": 20}]
    u = render_refinement(ctx, round_index=3, prev=prev, best_valid=best, history=history, limits=study["feedback"])
    assert "[line withheld]" in u and "roofline" not in u and "speedup" not in u
    assert "valid on 7 of 20 cases" in u and "first failing case: n=3145728" in u
    assert "geometric mean 1.5000 ms" in u and "| 1 | 20/20 | 1.5000 |" in u
    assert "Best valid candidate so far (round 1" in u
    fb.assert_feedback_clean(u.split("# Optimization round")[1])


def test_violation_round_feeds_its_diagnostic_to_the_next_round_and_no_repair_prompt_exists(study, folds):
    # protocol revision 3: a confirmed violation closes its round; the NEXT round's prompt carries the diagnostic
    # and the last compliant implementation; there is no same-round repair template any more
    assert not (TEMPLATE_DIR / "compliance_repair.md").exists()
    ctx, _, _ = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    prev = {"round": 2, "status": "contract_violation", "source": None, "configs_distinct": None, "latency_ms_geomean": None,
            "valid_cases": None, "cases_total": None, "diagnostic": "line 3: autotune decorator triton.autotune"}
    u = render_refinement(ctx, round_index=3, prev=prev, best_valid=None, history=[], limits=study["feedback"],
                          fallback={"round": 1, "source": "good"})
    assert "Optimization round 3 of 5" in u and "autotune decorator triton.autotune" in u
    assert "Last compliant implementation (round 1)" in u and "rejected for a contract violation" in u
    assert "attempt" not in u.split("# Optimization round")[1].lower()


def test_sanitize_truncates_by_fixed_rule():
    text = "\n".join(f"line {i}" for i in range(100))
    out = fb.sanitize_diagnostic(text, max_lines=5, max_chars=1000)
    assert out.count("\n") == 5 and "truncated to 5 lines" in out
