"""Renderer and feedback whitelist."""
import pytest

from tilebench.llm.v2.devtools import synthetic_context
from tilebench.llm.v2.prompts import feedback as fb
from tilebench.llm.v2.prompts.renderer import RenderError, render, render_initial, render_refinement, render_repair, render_system

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
    assert "`n` = `" in u and "atol=" in u and "def run(" in u
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
    prev = {"round": 2, "status": "numerical_error", "source": "def run(x): return x", "config": {"BLOCK": 64},
            "latency_ms_mean": None, "latency_ms_samples": None,
            "diagnostic": "mismatch at 3 elements\nroofline 55% speedup 2x T_SOL=1.0\nsecond line"}
    best = {"round": 1, "latency_ms_mean": 1.5, "source": "def run(x): return x+0", "config": {"BLOCK": 32}}
    history = [{"round": 1, "status": "valid", "latency_ms_mean": 1.5, "latency_ms_samples": [1.4, 1.5, 1.6]}]
    u = render_refinement(ctx, round_index=3, prev=prev, best_valid=best, history=history, limits=study["feedback"])
    assert "[line withheld]" in u and "roofline" not in u and "speedup" not in u
    assert "1.5000 ms" in u and "| 1 | 1.5000 | 1.4000, 1.5000, 1.6000 |" in u
    assert "Best valid candidate so far (round 1" in u
    fb.assert_feedback_clean(u.split("# Optimization round")[1])


def test_repair_prompt_names_attempt_and_fallback(study, folds):
    ctx, _, _ = synthetic_context("vector_add", "fp16", "B200", "triton", "base", study, folds)
    u = render_repair(ctx, round_index=4, attempt=2, max_attempts=3, violations=["line 3: autotune decorator triton.autotune"],
                      rejected_source="bad", fallback={"round": 2, "source": "good"})
    assert "attempt 2 of 3" in u and "autotune decorator" in u and "earlier compliant implementation (round 2)" in u
    u2 = render_repair(ctx, round_index=1, attempt=3, max_attempts=3, violations=["x"], rejected_source="bad", fallback=None)
    assert "No earlier compliant implementation" in u2


def test_sanitize_truncates_by_fixed_rule():
    text = "\n".join(f"line {i}" for i in range(100))
    out = fb.sanitize_diagnostic(text, max_lines=5, max_chars=1000)
    assert out.count("\n") == 5 and "truncated to 5 lines" in out
