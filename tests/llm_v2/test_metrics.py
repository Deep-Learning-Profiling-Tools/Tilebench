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


def test_cumulative_cost_sums_all_attempts_including_violations():
    costs, known = e.cumulative_costs(rounds(([10], True, 1.0), ([5, 6, 7], False, None), ([2], True, 0.5)))
    assert costs == [10, 28, 30] and known == 3


def test_more_than_three_attempts_is_an_error():
    with pytest.raises(ValueError):
        e.cumulative_costs(rounds(([1, 1, 1, 1], False, None)))


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
    curves = {("op1", "fp16"): e.curve(1.0, rounds(([10], True, 1.0))),
              ("op1", "fp32"): e.curve(1.0, rounds(([10], False, None))),
              ("op2", "fp16"): e.curve(1.0, rounds(([10], True, 2.0))),
              ("op3", "fp16"): None,
              ("op4", "fp16"): e.curve(1.0, rounds(([10], True, 1.0)))}
    elig = {("op1", "fp16"): "eligible", ("op1", "fp32"): "eligible", ("op2", "fp16"): "eligible",
            ("op3", "fp16"): "unsupported", ("op4", "fp16"): "incomplete"}
    agg = e.aggregate(curves, [10], elig)
    # op1 mean = (1.0 + 0)/2 = 0.5 ; op2 = 0.5 ; op3 excluded ; op4 incomplete excluded
    assert agg["mean"] == [0.5] and agg["n_operators"] == 2
    assert agg["unsupported"] == [("op3", "fp16")] and agg["incomplete"] == [("op4", "fp16")] and agg["partial"]


def test_t_sol_uses_declared_mode_and_flags_missing_peak():
    modes = ms.load_arithmetic_modes()
    metrics = {"flops_expr": "2 * M * N * K", "bytes_expr": "(M*K + K*N + M*N) * dtype_size"}
    r = t_sol("X", "batched_matmul", "fp16", {"M": 1024, "N": 1024, "K": 1024}, 1, metrics, modes, PEAK)
    assert r.status == "ok" and r.arithmetic_mode == "tc_fp16" and r.t_sol_ms == max(r.compute_term_ms, r.memory_term_ms)
    r2 = t_sol("X", "relu", "fp32", {"n": 1 << 20}, 1 << 20, {"flops_expr": "n", "bytes_expr": "2*n*dtype_size"}, modes, PEAK)
    assert r2.status == "peak_missing" and r2.arithmetic_mode == "fp32_vector" and r2.t_sol_ms is None
    r3 = t_sol("X", "batched_matmul", "fp32", {"M": 64, "N": 64, "K": 64}, 1, metrics, modes, PEAK)
    assert r3.arithmetic_mode == "tf32" and r3.p_peak_tflops == 50.0      # B200-style table: fp32 entry is TF32


def test_b200_peak_table_maps_fp32_to_tf32_only():
    from tilebench.core.metrics import load_peak_config
    from tilebench.llm.v2.metrics.sol import peak_for_mode
    peak = load_peak_config("B200")
    assert peak_for_mode(peak, "tf32") == 1100 and peak_for_mode(peak, "fp32_vector") is None
