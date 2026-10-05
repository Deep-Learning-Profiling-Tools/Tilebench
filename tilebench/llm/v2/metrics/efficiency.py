"""SOL-Efficiency@B and its aggregation, exactly as in the protocol:

    C_i   = sum_{j<=i} sum_{a<=A_j} c_{j,a},   A_j <= 3
    E(B)  = max({T_SOL / T_i : C_i <= B and valid_i = 1} union {0})
    E_bar(B) = mean_over_operators(mean_over_eligible_dtypes(E_{o,d}(B)))

Inputs are round records: {"round": i, "attempts": [{"cost": c or None}],
"valid": bool, "latency_ms": T_i or None}. Costs are logical tokens from the
ledger. A round whose cost is unknown makes every later C_i unknown: the
curve is then exact only up to the last fully-known round, which is reported
as `cost_known_through_round`."""
from __future__ import annotations

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
    best_efficiency: float
    audit_flags: list[str] = field(default_factory=list)
    status: str = "complete"          # complete | incomplete | unsupported | sol_unavailable

    def to_dict(self) -> dict:
        return {"t_sol_ms": self.t_sol_ms, "points": [asdict(p) for p in self.points],
                "cost_known_through_round": self.cost_known_through_round,
                "best_efficiency": self.best_efficiency, "audit_flags": self.audit_flags, "status": self.status}


def cumulative_costs(rounds: list[dict]) -> tuple[list[int | None], int]:
    """C_i per round; None from the first round with an unknown attempt cost."""
    out: list[int | None] = []
    total = 0
    known_through = 0
    unknown = False
    for r in rounds:
        attempts = r.get("attempts", [])
        if len(attempts) > 3:
            raise ValueError(f"round {r.get('round')} has {len(attempts)} attempts (> 3)")
        for a in attempts:
            c = a.get("cost")
            if c is None:
                unknown = True
            elif not unknown:
                total += int(c)
        if unknown:
            out.append(None)
        else:
            out.append(total)
            known_through = int(r.get("round", len(out)))
    return out, known_through


def curve(t_sol_ms: float | None, rounds: list[dict], *, status: str = "complete") -> TrajectoryCurve:
    costs, known = cumulative_costs(rounds)
    points: list[RoundPoint] = []
    best = 0.0
    flags: list[str] = []
    for r, c in zip(rounds, costs):
        valid = bool(r.get("valid")) and r.get("latency_ms") is not None and r["latency_ms"] > 0
        eff = None
        if valid and t_sol_ms is not None:
            eff = t_sol_ms / float(r["latency_ms"])
            if eff > 1.0:
                flags.append(f"round {r.get('round')}: efficiency {eff:.3f} > 1 (audit measurement/model; not an automatic hacking verdict)")
            best = max(best, eff)
        points.append(RoundPoint(round=int(r.get("round", len(points) + 1)), cumulative_cost=c, valid=valid,
                                 latency_ms=r.get("latency_ms"), efficiency=eff))
    if t_sol_ms is None:
        status = "sol_unavailable"
    return TrajectoryCurve(t_sol_ms=t_sol_ms, points=points, cost_known_through_round=known,
                           best_efficiency=best if t_sol_ms is not None else 0.0, audit_flags=flags, status=status)


def efficiency_at(curve_: TrajectoryCurve, budget: int) -> float | None:
    """E(B): best efficiency among valid rounds with C_i <= B. None when the
    answer is not determinable (a round with unknown cost could lie under B)."""
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


def aggregate(task_curves: dict[tuple[str, str], TrajectoryCurve | None], budgets: Iterable[int],
              eligible: dict[tuple[str, str], str]) -> dict:
    """Operator-balanced E_bar(B).

    task_curves: {(operator, dtype): curve or None}; eligible: {(op, dtype): status}
    where status in {eligible, unsupported, incomplete}. Eligible tasks without
    a curve or without a valid round count as 0. Unsupported tasks are excluded
    and listed; incomplete tasks are listed and excluded from the mean (the
    mean is then flagged as partial)."""
    budgets = list(budgets)
    by_op: dict[str, dict[str, TrajectoryCurve | None]] = {}
    unsupported, incomplete, undetermined = [], [], []
    for (op, dt), status in eligible.items():
        if status == "unsupported":
            unsupported.append((op, dt))
            continue
        if status == "incomplete":
            incomplete.append((op, dt))
            continue
        by_op.setdefault(op, {})[dt] = task_curves.get((op, dt))
    result = {"budgets": budgets, "mean": [], "per_operator": {}, "unsupported": unsupported,
              "incomplete": incomplete, "n_operators": len(by_op), "partial": bool(incomplete)}
    for B in budgets:
        op_means = []
        for op, dts in by_op.items():
            vals = []
            for dt, c in dts.items():
                if c is None:
                    vals.append(0.0)
                    continue
                e = efficiency_at(c, B)
                if e is None:
                    undetermined.append((op, dt, B))
                    e = 0.0
                vals.append(e)
            m = sum(vals) / len(vals)
            op_means.append(m)
            result["per_operator"].setdefault(op, []).append(m)
        result["mean"].append(sum(op_means) / len(op_means) if op_means else 0.0)
    result["undetermined"] = undetermined
    return result


def paired_difference(enhanced: dict, base: dict) -> list[float]:
    if enhanced["budgets"] != base["budgets"]:
        raise ValueError("paired comparison needs identical budget grids")
    return [e - b for e, b in zip(enhanced["mean"], base["mean"])]
