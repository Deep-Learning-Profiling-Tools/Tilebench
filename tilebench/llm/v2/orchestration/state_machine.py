"""Pure transition logic of the five-round protocol (study revision 4).

Round rules (study.yaml trajectory.round_rule):
- One round == one candidate-generation attempt. There is no same-round
  regeneration of any kind.
- A cleared (`clear`) or audit-flagged (`audit_only`) candidate is evaluated:
  verification, then timing.
- A confirmed contract violation (static, or confirmed by the evaluator at
  execution time: autotuner object found, output aliases an input, output
  cached across calls, precision state changed) closes the round as
  `contract_violation`; its diagnostic reaches the next round's prompt.
- Format, interface, compile, runtime, numerical and timing failures close
  the round.
- A valid but slower candidate is recorded; best_valid is never regressed.
- review_required holds the trajectory until the supervising Claude Code
  session decides (reviewer `claude-code`): `compliant` evaluates the archived
  candidate (no new request); `violation` closes the round as
  contract_violation.
- Every trajectory runs all rounds; there is no performance early stop.
- Transport failures past the retry budget, and provider refusals (auth /
  parameter errors), mark the trajectory incomplete (resumable); nothing is
  fabricated. Transport retries are never candidate attempts.
- Revision 4: the candidate is evaluated on every case of the task's frozen
  case suite; the round is valid only when every case is valid, each with
  three finite positive samples whose arithmetic mean is the case mean; the
  round latency is the geometric mean of the case means (recomputed here)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from tilebench.llm.v2.orchestration.state import AttemptRecord, RoundRecord, TrajectoryState

EVAL_STATUS_FROM_WORKER = {
    "valid": "valid", "interface_error": "interface_error", "compile_error": "compile_error",
    "runtime_error": "runtime_error", "numerical_error": "numerical_error", "timing_error": "timing_error",
    "contract_violation": "contract_violation", "infrastructure_incomplete": "infrastructure_incomplete",
}


@dataclass
class Action:
    kind: str            # generate_initial | repair | evaluate | done | blocked_review | incomplete
    round: int = 0
    attempt: int = 0


def current_round(state: TrajectoryState) -> RoundRecord | None:
    if state.rounds and state.rounds[-1].status == "pending":
        return state.rounds[-1]
    return None


EVALUABLE_VERDICTS = ("clear", "audit_only")


def next_action(state: TrajectoryState, *, rounds: int = 5, max_generations: int = 1) -> Action:
    if max_generations != 1:
        raise ValueError("protocol revision 3 allows exactly one candidate generation per round")
    if state.status == "review_required":
        return Action("blocked_review", round=state.rounds[-1].round if state.rounds else 0)
    if state.status == "incomplete":
        return Action("incomplete", round=state.rounds[-1].round if state.rounds else 0)
    done_rounds = [r for r in state.rounds if r.status != "pending"]
    cur = current_round(state)
    if cur is None:
        if len(done_rounds) >= rounds:
            return Action("done")
        return Action("generate_initial", round=len(done_rounds) + 1, attempt=1)
    if not cur.attempts:
        return Action("generate_initial", round=cur.round, attempt=1)
    last = cur.attempts[-1]
    if last.verdict in EVALUABLE_VERDICTS:
        return Action("evaluate", round=cur.round, attempt=last.attempt)
    if last.verdict == "review_required":
        return Action("blocked_review", round=cur.round, attempt=last.attempt)
    raise RuntimeError(f"unexpected pending state: verdict {last.verdict!r} (a round with one attempt must have been closed)")


def open_round(state: TrajectoryState, round_index: int) -> RoundRecord:
    if state.status == "complete":
        raise RuntimeError("trajectory is complete; no round beyond the protocol's rounds is opened")
    cur = current_round(state)
    if cur is not None:
        if cur.round != round_index:
            raise RuntimeError(f"round {cur.round} is still pending")
        return cur
    if any(r.round == round_index for r in state.rounds):
        raise RuntimeError(f"round {round_index} already closed")
    rec = RoundRecord(round=round_index)
    state.rounds.append(rec)
    return rec


def apply_attempt(state: TrajectoryState, round_index: int, attempt: AttemptRecord, *,
                  max_generations: int = 1, rounds: int = 5) -> None:
    rec = open_round(state, round_index)
    if attempt.attempt != len(rec.attempts) + 1:
        raise RuntimeError(f"attempt {attempt.attempt} out of order (have {len(rec.attempts)})")
    if attempt.attempt > max_generations or max_generations != 1:
        raise RuntimeError("protocol revision 3: exactly one candidate generation per round")
    rec.attempts.append(attempt)
    if attempt.verdict == "format_error":
        rec.status, rec.diagnostic = "format_error", attempt.diagnostic
    elif attempt.verdict == "transport_failed":
        rec.status, rec.diagnostic = "infrastructure_incomplete", attempt.error
        state.status = "incomplete"
        state.stop_reason = state.stop_reason or "transport retries exhausted"
    elif attempt.verdict == "provider_refused":
        rec.status, rec.diagnostic = "infrastructure_incomplete", attempt.error
        state.status = "incomplete"
        state.stop_reason = state.stop_reason or f"provider refused the request: {attempt.error}"
    elif attempt.verdict == "review_required":
        rec.status = "review_required"
        state.status = "review_required"
    elif attempt.verdict == "confirmed_violation":
        rec.status = "contract_violation"
        rec.diagnostic = attempt.diagnostic
    # clear / audit_only leave the round pending until evaluation.
    _close_if_finished(state, rounds)


def _valid_samples(result: dict) -> tuple[bool, str]:
    samples = result.get("latency_ms_samples")
    mean = result.get("latency_ms_mean")
    if not isinstance(samples, list) or len(samples) != 3:
        return False, "timing did not yield three samples"
    try:
        s = [float(x) for x in samples]
        m = float(mean)
    except (TypeError, ValueError):
        return False, "timing samples/mean are not numbers"
    if any(not math.isfinite(x) or x <= 0 for x in s) or not math.isfinite(m) or m <= 0:
        return False, "timing did not yield three finite positive samples"
    if abs(m - sum(s) / 3) > 1e-9 * max(1.0, m):
        return False, "reported mean is not the arithmetic mean of the three samples"
    return True, ""


def _as_suite(result: dict) -> dict:
    """A legacy single-case worker result (no `cases`) as a one-case suite."""
    if "cases" in result:
        return result
    case = {"case_id": "single", "case_index": 0, "status": result.get("status"), "latency_ms_mean": result.get("latency_ms_mean"),
            "latency_ms_samples": result.get("latency_ms_samples"), "config": result.get("config"),
            "timing": result.get("timing"), "timing_mode_differs": result.get("timing_mode_differs"),
            "diagnostic": result.get("diagnostic")}
    return {**result, "cases": [case], "cases_total": 1, "valid_cases": int(result.get("status") == "valid"),
            "latency_ms_geomean": result.get("latency_ms_mean") if result.get("status") == "valid" else None}


def suite_check(result: dict) -> tuple[bool, str, float | None]:
    """(ok, reason, recomputed geometric mean) of a suite whose worker status is valid."""
    cases = result.get("cases") or []
    total = result.get("cases_total")
    if not cases or total != len(cases):
        return False, f"case suite incomplete: {len(cases)} records for {total} cases", None
    logs = []
    for c in cases:
        if c.get("status") != "valid":
            return False, f"case {c.get('case_id')} is {c.get('status')}; a valid round needs every case valid", None
        ok, why = _valid_samples(c)
        if not ok:
            return False, f"case {c.get('case_id')}: {why}", None
        logs.append(math.log(float(c["latency_ms_mean"])))
    gm = math.exp(sum(logs) / len(logs))
    rep = result.get("latency_ms_geomean")
    if rep is not None and abs(float(rep) - gm) > 1e-9 * max(1.0, gm):
        return False, "reported geometric mean is not the geometric mean of the case means", None
    return True, "", gm


def compact_cases(result: dict) -> list[dict]:
    keep = ("case_id", "case_index", "status", "latency_ms_mean", "latency_ms_samples", "config", "timing_mode_differs")
    out = []
    for c in result.get("cases") or []:
        row = {k: c.get(k) for k in keep}
        row["timing_execution_mode"] = (c.get("timing") or {}).get("timing_execution_mode")
        if c.get("status") not in ("valid", "not_evaluated"):
            row["diagnostic"] = str(c.get("diagnostic") or "")[:600]
        out.append(row)
    return out


def compact_evaluation(result: dict) -> dict:
    """What trajectory.json keeps of a suite result (the full record is eval_NNNN/evaluation.json)."""
    keep = ("schema", "status", "cases_total", "cases_evaluated", "valid_cases", "latency_ms_geomean", "first_failing_case",
            "suite_stopped", "timing_mode_differs", "configs_distinct", "worker_wall_s", "device_lock_wait_s", "worker_rc",
            "evaluation_revision", "seed", "elapsed_s")
    out = {k: result.get(k) for k in keep if k in result}
    out["diagnostic"] = str(result.get("diagnostic") or "")[:4000] or None
    res = result.get("resources") or {}
    if res:
        out["cgroup_oom_kill_delta"] = res.get("cgroup_oom_kill_delta")
    out["isolation_backend"] = (result.get("isolation") or {}).get("backend")
    return out


def apply_evaluation(state: TrajectoryState, round_index: int, result: dict, *, max_generations: int = 1,
                     rounds: int = 5) -> None:
    """result: worker suite output {status, cases[...], cases_total, valid_cases, latency_ms_geomean, diagnostic, ...}
    (a legacy single-case result is treated as a one-case suite)."""
    rec = open_round(state, round_index)
    if not rec.attempts or rec.attempts[-1].verdict not in EVALUABLE_VERDICTS:
        raise RuntimeError("evaluation requires a clear or audit_only attempt")
    result = _as_suite(result)
    status = EVAL_STATUS_FROM_WORKER.get(result.get("status"), "runtime_error")
    last = rec.attempts[-1]
    rec.evaluation = compact_evaluation(result)
    rec.case_results = compact_cases(result)
    rec.cases_total = result.get("cases_total")
    rec.valid_cases = sum(1 for c in rec.case_results if c.get("status") == "valid")
    rec.timing_mode_differs = bool(result.get("timing_mode_differs"))
    modes = sorted({c.get("timing_execution_mode") for c in rec.case_results if c.get("timing_execution_mode")})
    rec.timing_execution_mode = modes[0] if len(modes) == 1 else ("mixed:" + "+".join(modes) if modes else None)
    if status == "contract_violation":
        # execution-confirmed violation: the round closes, no regeneration
        last.verdict = "confirmed_violation"
        last.diagnostic = result.get("diagnostic")
        last.execution = {k: result.get(k) for k in ("status", "diagnostic", "first_failing_case", "suite_stopped", "seed", "archive")}
        rec.status, rec.diagnostic = "contract_violation", result.get("diagnostic")
        _close_if_finished(state, rounds)
        return
    rec.status = status
    rec.source_path = last.source_path
    rec.config = {"configs_distinct": result.get("configs_distinct") or []}
    rec.diagnostic = result.get("diagnostic")
    if status == "valid":
        ok, why, gm = suite_check(result)
        if not ok:
            rec.status, rec.diagnostic = "timing_error", why
        else:
            rec.latency_ms_geomean = gm
            if state.best_valid is None or gm < state.best_valid["latency_ms_geomean"]:
                state.best_valid = {"round": rec.round, "latency_ms_geomean": gm, "source_path": rec.source_path,
                                    "configs_distinct": result.get("configs_distinct") or [],
                                    "timing_execution_mode": rec.timing_execution_mode}
    elif status == "infrastructure_incomplete":
        state.status = "incomplete"
        state.stop_reason = state.stop_reason or f"evaluation infrastructure: {result.get('diagnostic')}"
    _close_if_finished(state, rounds)


def resolve_review(state: TrajectoryState, round_index: int, decision: str, note: str, *, reviewer: str = "claude-code",
                   rounds: int = 5, record: dict | None = None) -> None:
    """Compliance decision for a review_required round (frozen-contract
    compliance only, never performance): 'compliant' re-opens the round so the
    ARCHIVED candidate is evaluated (no new request); 'violation' closes the
    round as contract_violation (no regeneration; the next round receives the
    diagnostic). `record` (hashes, evidence, rationale, timestamp) is appended
    to the round's reviews."""
    rec = next(r for r in state.rounds if r.round == round_index)
    if rec.status != "review_required":
        raise RuntimeError("round is not awaiting review")
    last = rec.attempts[-1]
    rec.reviews.append({"reviewer": reviewer, "decision": decision, "note": note, **(record or {})})
    state.notes.append(f"round {round_index} attempt {last.attempt}: review by {reviewer} -> {decision}: {note}")
    if decision == "compliant":
        last.verdict = "clear"
        rec.status = "pending"
        state.status = "in_progress"
    elif decision == "violation":
        last.verdict = "confirmed_violation"
        rec.status = "contract_violation"
        rec.diagnostic = (f"contract violation (compliance review by {reviewer}): {note}" if note else last.diagnostic)
        state.status = "in_progress"
        _close_if_finished(state, rounds)
    else:
        raise ValueError(decision)


def _close_if_finished(state: TrajectoryState, rounds: int = 5) -> None:
    if state.status in ("incomplete", "review_required"):
        return
    done = [r for r in state.rounds if r.status != "pending"]
    if len(done) >= rounds and current_round(state) is None:
        state.status = "complete"
