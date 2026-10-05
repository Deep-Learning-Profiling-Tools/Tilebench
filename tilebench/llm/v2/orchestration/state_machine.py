"""Pure transition logic of the ten-round protocol.

Round rules (study §3.3):
- A round starts with one generation. Only a CONFIRMED contract violation or
  reward-hacking attempt triggers a same-round regeneration, up to 3
  generations in total (initial + 2 repairs). A violation confirmed by the
  evaluator at execution time (autotuner object found, output aliases an
  input) is a confirmed violation too: the attempt's verdict is rewritten
  and the repair path applies.
- The first candidate that clears compliance is evaluated; no further
  candidates are generated in that round.
- Format, interface, compile, runtime, numerical and timing failures consume
  the round.
- Three violations: the round has no valid candidate; all costs stay charged.
- A valid but slower candidate is recorded; best_valid is never regressed.
- review_required blocks the trajectory until a human decision.
- Transport failures past the retry budget, and provider refusals (auth /
  parameter errors), mark the trajectory incomplete (resumable); nothing is
  fabricated.
- A valid round needs three finite positive samples whose arithmetic mean
  equals the reported mean."""
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


def next_action(state: TrajectoryState, *, rounds: int = 10, max_generations: int = 3) -> Action:
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
    if last.verdict == "clear":
        return Action("evaluate", round=cur.round, attempt=last.attempt)
    if last.verdict == "confirmed_violation":
        if len(cur.attempts) < max_generations:
            return Action("repair", round=cur.round, attempt=len(cur.attempts) + 1)
        raise RuntimeError("round with 3 violations must have been closed by apply_attempt")
    if last.verdict == "review_required":
        return Action("blocked_review", round=cur.round, attempt=last.attempt)
    raise RuntimeError(f"unexpected pending state: verdict {last.verdict!r}")


def open_round(state: TrajectoryState, round_index: int) -> RoundRecord:
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
                  max_generations: int = 3) -> None:
    rec = open_round(state, round_index)
    if attempt.attempt != len(rec.attempts) + 1:
        raise RuntimeError(f"attempt {attempt.attempt} out of order (have {len(rec.attempts)})")
    if attempt.attempt > max_generations:
        raise RuntimeError("more than 3 generations in one round")
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
    elif attempt.verdict == "confirmed_violation" and len(rec.attempts) >= max_generations:
        rec.status = "contract_violation"
        rec.diagnostic = attempt.diagnostic
    # "clear" leaves the round pending until evaluation; a violation with
    # attempts left also leaves it pending (next action: repair).
    _close_if_finished(state)


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


def apply_evaluation(state: TrajectoryState, round_index: int, result: dict, *, max_generations: int = 3) -> None:
    """result: worker output {status, latency_ms_mean, latency_ms_samples, diagnostic, ...}."""
    rec = open_round(state, round_index)
    if not rec.attempts or rec.attempts[-1].verdict != "clear":
        raise RuntimeError("evaluation requires a cleared attempt")
    status = EVAL_STATUS_FROM_WORKER.get(result.get("status"), "runtime_error")
    last = rec.attempts[-1]
    if status == "contract_violation":
        # execution-confirmed violation: same-round repair rules apply
        last.verdict = "confirmed_violation"
        last.diagnostic = result.get("diagnostic")
        last.execution = {k: result.get(k) for k in ("status", "diagnostic", "stages", "isolation", "seed", "archive")}
        if len(rec.attempts) >= max_generations:
            rec.status, rec.diagnostic = "contract_violation", result.get("diagnostic")
        _close_if_finished(state)
        return
    rec.evaluation = result
    rec.status = status
    rec.source_path = last.source_path
    rec.config = result.get("config") or last.config
    rec.diagnostic = result.get("diagnostic")
    timing = result.get("timing") or {}
    rec.timing_execution_mode = timing.get("timing_execution_mode")
    rec.timing_mode_differs = result.get("timing_mode_differs")
    if status == "valid":
        ok, why = _valid_samples(result)
        if not ok:
            rec.status, rec.diagnostic = "timing_error", why
        else:
            samples = [float(x) for x in result["latency_ms_samples"]]
            mean = float(result["latency_ms_mean"])
            rec.latency_ms_mean, rec.latency_ms_samples = mean, samples
            if state.best_valid is None or mean < state.best_valid["latency_ms_mean"]:
                state.best_valid = {"round": rec.round, "latency_ms_mean": mean,
                                    "source_path": rec.source_path, "config": rec.config,
                                    "timing_execution_mode": rec.timing_execution_mode}
    elif status == "infrastructure_incomplete":
        state.status = "incomplete"
        state.stop_reason = state.stop_reason or f"evaluation infrastructure: {result.get('diagnostic')}"
    _close_if_finished(state)


def resolve_review(state: TrajectoryState, round_index: int, decision: str, note: str, *, reviewer: str = "human") -> None:
    """Human decision for a review_required round: 'compliant' re-opens the
    round for evaluation; 'violation' counts as a confirmed violation."""
    rec = next(r for r in state.rounds if r.round == round_index)
    if rec.status != "review_required":
        raise RuntimeError("round is not awaiting review")
    last = rec.attempts[-1]
    state.notes.append(f"round {round_index} attempt {last.attempt}: review by {reviewer} -> {decision}: {note}")
    if decision == "compliant":
        last.verdict = "clear"
        rec.status = "pending"
        state.status = "in_progress"
    elif decision == "violation":
        last.verdict = "confirmed_violation"
        rec.status = "pending"
        state.status = "in_progress"
        if len(rec.attempts) >= 3:
            rec.status = "contract_violation"
            rec.diagnostic = last.diagnostic or note
        _close_if_finished(state)
    else:
        raise ValueError(decision)


def _close_if_finished(state: TrajectoryState, rounds: int = 10) -> None:
    if state.status in ("incomplete", "review_required"):
        return
    done = [r for r in state.rounds if r.status != "pending"]
    if len(done) >= rounds and current_round(state) is None:
        state.status = "complete"
