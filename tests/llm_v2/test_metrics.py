"""T_SOL, cumulative cost, E(B), aggregation."""
import pytest

from tilebench.llm.v2.manifests import schema as ms
from tilebench.llm.v2.metrics import efficiency as e
from tilebench.llm.v2.metrics.sol import t_sol

PEAK = {"peak_bw_GBs": 1000.0, "peak_tflops": {"fp16": 100.0, "fp32": 50.0}}


def rounds(*specs):
    out = []
    for i, (costs, valid, lat) in enumerate(specs, 1):
        out.append({"round": i, "attempts": [{"cost": c} for c in costs], "valid": valid, "latency_ms": lat})
    return out


def test_cumulative_cost_counts_every_sent_generation_whatever_its_outcome():
    # protocol revision 3: one generation per round; a failed / violating round's tokens are charged too
    costs, known = e.cumulative_costs(rounds(([10], True, 1.0), ([18], False, None), ([2], True, 0.5)))
    assert costs == [10, 28, 30] and known == 3


def test_more_than_one_attempt_per_round_is_an_error():
    with pytest.raises(ValueError):
        e.cumulative_costs(rounds(([1, 1], False, None)))
    # revision-2 pilot states (up to 3 attempts) are readable only when the caller says so explicitly
    assert e.cumulative_costs(rounds(([1, 1, 1], False, None)), max_attempts=3)[0] == [3]


def test_efficiency_zero_without_valid_and_best_never_erased():
    c = e.curve(1.0, rounds(([10], False, None), ([10], True, 2.0), ([10], False, None), ([10], True, 4.0)))
    assert e.efficiency_at(c, 5) == 0.0
    assert e.efficiency_at(c, 20) == 0.5
    assert e.efficiency_at(c, 40) == 0.5                 # slower valid round 4 does not lower it
    assert e.step_curve(c) == [(10, 0.0), (20, 0.5), (30, 0.5), (40, 0.5)]


def test_ratio_above_one_kept_and_flagged():
    c = e.curve(2.0, rounds(([10], True, 1.0)))
    assert c.best_efficiency == 2.0 and c.audit_flags and "not an automatic hacking verdict" in c.audit_flags[0]


def test_unknown_cost_makes_later_budget_points_undetermined():
    c = e.curve(1.0, rounds(([10], True, 2.0), ([None], True, 1.0), ([10], True, 1.0)))
    assert c.cost_known_through_round == 1
    assert e.efficiency_at(c, 10) == 0.5
    assert e.efficiency_at(c, 100) is None


def test_non_uniform_token_steps_not_round_indices():
    a = e.curve(1.0, rounds(([100], True, 1.0)))
    b = e.curve(1.0, rounds(([10], True, 1.0)))
    assert e.efficiency_at(a, 50) == 0.0 and e.efficiency_at(b, 50) == 1.0


def test_operator_balanced_aggregation_with_zero_unsupported_incomplete():
    # one representative dtype per operator: E_bar(B) is the mean over the predeclared operators
    curves = {("op1", "fp16"): e.curve(1.0, rounds(([10], False, None))),
              ("op2", "fp16"): e.curve(1.0, rounds(([10], True, 2.0))),
              ("op3", "fp16"): None,
              ("op4", "fp16"): e.curve(1.0, rounds(([10], True, 1.0)), status="incomplete"),
              ("op5", "fp32"): e.curve(1.0, rounds(([10], True, 1.0)))}
    elig = {("op1", "fp16"): "eligible", ("op2", "fp16"): "eligible", ("op3", "fp16"): "unsupported",
            ("op4", "fp16"): "eligible", ("op5", "fp32"): "eligible"}        # execution state is NOT eligibility
    agg = e.aggregate(curves, [10], elig)
    # op1 real 0 ; op2 0.5 ; op3 unsupported (out of the denominator) ; op4 incomplete -> exact mean unknown ; op5 1.0
    assert agg["mean"] == [None] and agg["n_operators"] == 4 and agg["partial"]
    assert agg["lower_bound_mean"] == [(0.0 + 0.5 + 0.0 + 1.0) / 4] and agg["completed_only_mean"] == [(0.0 + 0.5 + 1.0) / 3]
    assert agg["unsupported"] == [("op3", "fp16")] and agg["incomplete"] == [("op4", "fp16")]
    assert agg["tasks_included"] == [("op1", "fp16"), ("op2", "fp16"), ("op4", "fp16"), ("op5", "fp32")]
    with pytest.raises(ValueError, match="pre-declared"):
        e.aggregate({}, [10], {("op4", "fp16"): "incomplete"})
    with pytest.raises(ValueError, match="one representative dtype"):
        e.aggregate({}, [10], {("op1", "fp16"): "eligible", ("op1", "fp32"): "eligible"})
    with pytest.raises(ValueError, match="outside the predeclared set"):
        e.aggregate({("op9", "fp16"): curves[("op2", "fp16")]}, [10], {("op1", "fp16"): "eligible"})


def test_legacy_t_sol_uses_v2_declarations_without_aliases():
    modes = ms.load_arithmetic_modes()
    metrics = {"flops_expr": "2 * M * N * K", "bytes_expr": "(M*K + K*N + M*N) * dtype_size"}
    r = t_sol("X", "batched_matmul", "fp16", {"M": 1024, "N": 1024, "K": 1024}, 1, metrics, modes, PEAK)
    assert r.status == "ok" and r.arithmetic_mode == "mma_fp16_f32acc" and r.t_sol_ms == max(r.compute_term_ms, r.memory_term_ms)
    r2 = t_sol("X", "softmax", "fp32", {"n_rows": 64, "n_cols": 64}, 1, {"flops_expr": "n_rows*n_cols", "bytes_expr": "2*n_rows*n_cols*dtype_size"}, modes, PEAK)
    assert r2.status == "peak_missing" and r2.arithmetic_mode == "fp32_fma_vector" and r2.t_sol_ms is None
    r3 = t_sol("X", "batched_matmul", "fp32", {"M": 64, "N": 64, "K": 64}, 1, metrics, modes, PEAK)
    assert r3.arithmetic_mode == "mma_tf32_f32acc" and r3.status == "peak_missing"   # fp32 entry is never read as TF32
    r4 = t_sol("X", "batched_matmul", "fp32", {"M": 64, "N": 64, "K": 64}, 1, metrics, modes, {**PEAK, "fp32_is_tf32": True})
    assert r4.p_peak_tflops == 50.0                                                  # only an explicit declaration aliases
    r5 = t_sol("X", "relu", "fp32", {"n": 1 << 20}, 1 << 20, {"flops_expr": "n", "bytes_expr": "2*n*dtype_size"}, modes, PEAK)
    assert r5.status == "memory_only" and r5.t_sol_ms == r5.memory_term_ms


def test_b200_peak_table_maps_fp32_to_tf32_only():
    from tilebench.core.metrics import load_peak_config
    from tilebench.llm.v2.metrics.sol import peak_for_mode
    peak = load_peak_config("B200")
    assert peak_for_mode(peak, "mma_tf32_f32acc") == 1100 and peak_for_mode(peak, "fp32_fma_vector") is None
    assert peak_for_mode({"peak_tflops": {"fp32": 50.0}}, "mma_tf32_f32acc") is None   # no implicit fp32->tf32
