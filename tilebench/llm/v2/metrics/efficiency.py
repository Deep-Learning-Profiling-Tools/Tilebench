"""SOL-Efficiency@B and its aggregation, exactly as in the protocol
(revision 4: 5 rounds, exactly one candidate generation per round, one
predeclared representative dtype per operator, every candidate evaluated on
the operator's 20 frozen cases):

    C_i   = sum_{j<=i} c_j,   i in {1..5}   (c_j: every sent generation of round j, whatever its outcome)
    eff_i = geomean_j (T_emp_j / T_ij)  over the 20 cases j of a valid round i (valid = 20/20 cases valid)
    E(B)  = max({eff_i : C_i <= B and valid_i = 1} union {0})
    E_bar(B) = arithmetic mean over the predeclared operators of E_{o,d(o)}(B)   (d(o): the operator's selected dtype)

`t_sol_ms` is either one target (single-case pilot rounds: eff_i = T_SOL / T_i)
or a {case_id: T_emp_j} mapping (revision 4: the round must report
`case_latency_ms` for every case). The same geometric-mean form gives the
reporting speedup with {case_id: T_torch_j} (`geomean_ratio`).

Two independent budget axes: E_token(B_token) with C_i in logical tokens
(the main metric) and E_usd(B_usd) with C_i in estimated list-price USD
(metrics.cost); the same curve logic, never mixed.

Inputs are round records: {"round": i, "attempts": [{"cost": c or None}],
"valid": bool, "latency_ms": T_i or None, "latency_ms_samples": [...]}.
Costs are logical tokens from the ledger.

What is a number and what is unknown:
- a complete trajectory whose valid rounds are all above the budget (or
  that has no valid round) has E(B) = 0: a real zero;
- a round whose cost is unknown makes every later C_i unknown; E(B) is then
  exact only for budgets up to the last fully-known cumulative cost
  (`cost_known_through_round`), and `None` (undetermined) beyond it
  whenever a later valid round could lie under B;
- a task without T_SOL (`sol_unavailable`), without a trajectory, or whose
  trajectory is incomplete has no E(B): `None`, never 0.
- a valid round needs a finite positive latency; when samples are present
  there must be three finite positive ones whose mean is the latency.

The aggregate therefore carries two series: `mean` (exact; None at any
budget where some included task is undetermined) and `lower_bound_mean`
(undetermined, missing and incomplete tasks counted as 0; labelled as a
conservative lower bound, never as E_bar). Rounds timed in a different
execution mode than the campaign expects are kept and listed in
`audit_flags` (timing_mode_differs)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable


@dataclass
class RoundPoint:
    round: int
    cumulative_cost: int | None
    valid: bool
    latency_ms: float | None
    efficiency: float | None          # T_SOL / T_i for valid rounds


@dataclass
class TrajectoryCurve:
    t_sol_ms: float | None
    points: list[RoundPoint]
    cost_known_through_round: int
    best_efficiency: float | None
    audit_flags: list[str] = field(default_factory=list)
    status: str = "complete"          # complete | incomplete | unsupported | sol_unavailable

    def to_dict(self) -> dict:
        return {"t_sol_ms": self.t_sol_ms, "points": [asdict(p) for p in self.points],
                "cost_known_through_round": self.cost_known_through_round,
                "best_efficiency": self.best_efficiency, "audit_flags": self.audit_flags, "status": self.status}


def _finite_positive(x) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0


def geomean_ratio(numerators: dict, case_latency_ms: dict | None) -> float | None:
    """geomean_j (numerators[j] / case_latency_ms[j]) over every case of `numerators`; None if any is missing."""
    if not numerators or not case_latency_ms:
        return None
    logs = []
    for cid, num in numerators.items():
        lat = case_latency_ms.get(cid)
        if not (_finite_positive(num) and _finite_positive(lat)):
            return None
        logs.append(math.log(float(num) / float(lat)))
    return math.exp(sum(logs) / len(logs))


def round_is_valid(r: dict) -> tuple[bool, str | None]:
    if not r.get("valid"):
        return False, None
    if r.get("cases_total"):
        lat = r.get("case_latency_ms") or {}
        if r.get("valid_cases") != r["cases_total"] or len(lat) != r["cases_total"]:
            return False, f"round {r.get('round')}: marked valid without every case valid ({r.get('valid_cases')}/{r.get('cases_total')})"
        if not all(_finite_positive(v) for v in lat.values()):
            return False, f"round {r.get('round')}: a case latency is not finite positive"
        for cid, samples in (r.get("case_samples_ms") or {}).items():
            if samples is not None and (len(samples) != 3 or not all(_finite_positive(x) for x in samples)
                                        or abs(float(lat[cid]) - sum(float(x) for x in samples) / 3) > 1e-9 * max(1.0, float(lat[cid]))):
                return False, f"round {r.get('round')}: case {cid} lacks three finite positive samples whose mean is its latency"
        return True, None
    lat = r.get("latency_ms")
    if not _finite_positive(lat):
        return False, f"round {r.get('round')}: marked valid but latency {lat!r} is not finite positive"
    samples = r.get("latency_ms_samples")
    if samples is not None:
        if len(samples) != 3 or not all(_finite_positive(s) for s in samples):
            return False, f"round {r.get('round')}: marked valid without three finite positive samples"
        if abs(float(lat) - sum(float(s) for s in samples) / 3) > 1e-9 * max(1.0, float(lat)):
            return False, f"round {r.get('round')}: latency is not the mean of its three samples"
    return True, None


def cumulative_costs(rounds: list[dict], *, max_attempts: int = 1, cost_key: str = "cost") -> tuple[list, int]:
    """C_i per round; None from the first round with an unknown attempt cost.
    Protocol revision 3 allows exactly one generation per round. cost_key
    "cost" = logical tokens (E_token), "usd" = estimated list-price USD
    (E_usd); the two axes are never mixed."""
    out: list[int | None] = []
    total = 0
    known_through = 0
    unknown = False
    for r in rounds:
        attempts = r.get("attempts", [])
        if len(attempts) > max_attempts:
            raise ValueError(f"round {r.get('round')} has {len(attempts)} attempts (> {max_attempts})")
        for a in attempts:
            c = a.get(cost_key)
            if c is None:
                unknown = True
            elif not unknown:
                total += int(c) if cost_key == "cost" else float(c)
        if unknown:
            out.append(None)
        else:
            out.append(total)
            known_through = int(r.get("round", len(out)))
    return out, known_through


def curve(t_sol_ms: float | None, rounds: list[dict], *, status: str = "complete", max_attempts: int = 1,
          cost_key: str = "cost") -> TrajectoryCurve:
    costs, known = cumulative_costs(rounds, max_attempts=max_attempts, cost_key=cost_key)
    points: list[RoundPoint] = []
    best: float | None = None
    flags: list[str] = []
    per_case = isinstance(t_sol_ms, dict)
    sol_ok = (bool(t_sol_ms) and all(_finite_positive(v) for v in t_sol_ms.values())) if per_case else \
        (t_sol_ms is not None and _finite_positive(t_sol_ms))
    for r, c in zip(rounds, costs):
        valid, why = round_is_valid(r)
        if why:
            flags.append(why)
        eff = None
        if valid and sol_ok and per_case:
            eff = geomean_ratio(t_sol_ms, r.get("case_latency_ms"))
            if eff is None:
                flags.append(f"round {r.get('round')}: case latencies do not cover the target's cases")
                valid = False
        elif valid and sol_ok:
            eff = float(t_sol_ms) / float(r["latency_ms"])
        if eff is not None:
            if eff > 1.0:
                flags.append(f"round {r.get('round')}: efficiency {eff:.3f} > 1 (audit measurement/model; not an automatic hacking verdict)")
            best = eff if best is None else max(best, eff)
        if valid and r.get("timing_mode_differs"):
            flags.append(f"round {r.get('round')}: timing_mode_differs (measured {r.get('timing_execution_mode')}; see capture_failure_policy)")
        points.append(RoundPoint(round=int(r.get("round", len(points) + 1)), cumulative_cost=c, valid=valid,
                                 latency_ms=r.get("latency_ms"), efficiency=eff))
    if not sol_ok:
        status = "sol_unavailable"
        best = None
    elif status == "complete" and best is None:
        best = 0.0
    return TrajectoryCurve(t_sol_ms=t_sol_ms, points=points, cost_known_through_round=known,
                           best_efficiency=best, audit_flags=flags, status=status)


def efficiency_at(curve_: TrajectoryCurve, budget: int) -> float | None:
    """E(B): best efficiency among valid rounds with C_i <= B. None when the
    answer is not determinable: no T_SOL, an incomplete trajectory, or a
    round with unknown cost that could lie under B."""
    if curve_.status in ("sol_unavailable", "incomplete", "unsupported"):
        return None
    best = 0.0
    last_known = 0
    for p in curve_.points:
        if p.cumulative_cost is None:
            # Every later round costs at least one more token than the last known
            # cumulative cost, so budgets up to it are still exact; beyond it the
            # answer is undetermined whenever a later valid round exists.
            if budget <= last_known:
                return best
            later_valid = any(q.valid and q.efficiency for q in curve_.points if q.round >= p.round)
            return None if later_valid else best
        last_known = p.cumulative_cost
        if p.cumulative_cost <= budget and p.valid and p.efficiency is not None:
            best = max(best, p.efficiency)
    return best


def step_curve(curve_: TrajectoryCurve) -> list[tuple[int, float]]:
    """(cost, best-so-far) breakpoints; a step function, no interpolation."""
    pts = []
    best = 0.0
    for p in curve_.points:
        if p.cumulative_cost is None:
            break
        if p.valid and p.efficiency is not None:
            best = max(best, p.efficiency)
        pts.append((p.cumulative_cost, best))
    return pts


PREDECLARED_STATUSES = ("eligible", "unsupported")


def aggregate(task_curves: dict[tuple[str, str], TrajectoryCurve | None], budgets: Iterable[int],
              eligible: dict[tuple[str, str], str], *, one_dtype_per_operator: bool = True) -> dict:
    """Operator-balanced E_bar(B) over the FROZEN pre-declared task set.

    Protocol revision 3: `eligible` holds exactly one (operator, selected
    dtype) per predeclared operator (one_dtype_per_operator=True refuses
    anything else), so E_bar(B) is the mean over the operators, with no
    inner dtype average.

    task_curves: {(operator, dtype): curve or None}; eligible: {(op, dtype):
    status} is the pre-declared eligibility (`eligible` | `unsupported`), fixed
    before generation. Execution state is never part of `eligible`: it is
    read from the curves (`complete` | `incomplete`) or from their absence.

    - `unsupported` tasks are excluded from the denominator and listed;
    - every `eligible` task stays in the denominator (`tasks_included`),
      whatever happened to it;
    - an eligible task without a curve (`missing`), with an `incomplete`
      trajectory, without T_SOL (`sol_unavailable`) or with an undetermined
      E(B) at a budget (unknown cost) makes the exact `mean` None at that
      budget; such tasks count as 0 only in `lower_bound_mean`;
    - only a complete trajectory with no valid round under B contributes a
      real 0 to the exact mean.
    `completed_only_mean` (coverage listed in `completed_only_tasks`) is the
    mean over complete tasks only: a differently named quantity, not E_bar(B)
    and not a substitute for the paired Base/Enhanced comparison."""
    budgets = list(budgets)
    if one_dtype_per_operator:
        per_op: dict[str, list[str]] = {}
        for op, dt in eligible:
            per_op.setdefault(op, []).append(dt)
        multi = {op: dts for op, dts in per_op.items() if len(dts) > 1}
        if multi:
            raise ValueError(f"one representative dtype per operator is required; got several for {multi}")
        stray = sorted(k for k in task_curves if k not in eligible)
        if stray:
            raise ValueError(f"curves for tasks outside the predeclared set: {stray}")
    for key, status in eligible.items():
        if status not in PREDECLARED_STATUSES:
            raise ValueError(f"eligibility of {key} must be pre-declared as one of {PREDECLARED_STATUSES}, got {status!r} "
                             "(execution state belongs to the curve, not to the eligibility table)")
    by_op: dict[str, dict[str, TrajectoryCurve | None]] = {}
    unsupported, incomplete, missing, sol_unavailable, undetermined = [], [], [], [], []
    for (op, dt), status in eligible.items():
        if status == "unsupported":
            unsupported.append((op, dt))
            continue
        c = task_curves.get((op, dt))
        if c is None:
            missing.append((op, dt))
        elif c.status == "incomplete":
            incomplete.append((op, dt))
        elif c.status == "sol_unavailable":
            sol_unavailable.append((op, dt))
        by_op.setdefault(op, {})[dt] = c
    tasks_included = sorted((op, dt) for op, dts in by_op.items() for dt in dts)
    completed_only_tasks = sorted((op, dt) for op, dts in by_op.items() for dt, c in dts.items()
                                  if c is not None and c.status == "complete")
    result = {"budgets": budgets, "mean": [], "lower_bound_mean": [], "completed_only_mean": [],
              "per_operator": {}, "per_operator_lower_bound": {}, "unsupported": unsupported,
              "incomplete": incomplete, "missing": missing, "sol_unavailable": sol_unavailable,
              "n_operators": len(by_op), "tasks_included": tasks_included,
              "completed_only_tasks": completed_only_tasks,
              "partial": bool(incomplete or missing or sol_unavailable)}
    for B in budgets:
        op_means: list[float | None] = []
        op_lower: list[float] = []
        op_completed: list[float] = []
        for op, dts in by_op.items():
            vals: list[float | None] = []
            lower: list[float] = []
            completed: list[float] = []
            for dt, c in dts.items():
                if c is None or c.status in ("sol_unavailable", "incomplete"):
                    vals.append(None)
                    lower.append(0.0)
                    continue
                e = efficiency_at(c, B)
                if e is None:
                    undetermined.append((op, dt, B))
                    vals.append(None)
                    lower.append(0.0)
                else:
                    vals.append(e)
                    lower.append(e)
                    completed.append(e)
            m = None if any(v is None for v in vals) else sum(vals) / len(vals)  # type: ignore[arg-type]
            op_means.append(m)
            op_lower.append(sum(lower) / len(lower))
            if completed:
                op_completed.append(sum(completed) / len(completed))
            result["per_operator"].setdefault(op, []).append(m)
            result["per_operator_lower_bound"].setdefault(op, []).append(op_lower[-1])
        exact = None if (not op_means or any(m is None for m in op_means)) else sum(op_means) / len(op_means)  # type: ignore[arg-type]
        result["mean"].append(exact)
        result["lower_bound_mean"].append(sum(op_lower) / len(op_lower) if op_lower else None)
        result["completed_only_mean"].append(sum(op_completed) / len(op_completed) if op_completed else None)
    result["undetermined"] = undetermined
    result["partial"] = result["partial"] or bool(undetermined)
    result["lower_bound_note"] = ("lower_bound_mean counts undetermined, missing, sol_unavailable and incomplete "
                                  "tasks as 0 over the full pre-declared denominator; it is a conservative bound, not E_bar(B)")
    result["completed_only_note"] = ("completed_only_mean averages complete trajectories only (coverage in completed_only_tasks); "
                                     "it is not E_bar(B) and must not replace the paired comparison")
    return result


def paired_difference(enhanced: dict, base: dict) -> list[float | None]:
    """Enhanced - Base per budget; requires the same budget grid and the same
    task coverage; None where either side is undetermined."""
    if enhanced["budgets"] != base["budgets"]:
        raise ValueError("paired comparison needs identical budget grids")
    if enhanced.get("tasks_included") != base.get("tasks_included"):
        raise ValueError("paired comparison needs identical task coverage")
    return [None if (e is None or b is None) else e - b for e, b in zip(enhanced["mean"], base["mean"])]
