"""Concurrent campaign scheduler: one process runs every (model, DSL) track of a
campaign (revision 4: GPT and Claude in the same process), each trajectory in
its own thread with the unchanged per-trajectory code (campaign.run_track_task
-> TrajectoryRunner). Trajectory semantics are not touched: 5 rounds, one
generation per round, same prompts, same model settings, same checker, same
scoring. Round i+1 of a trajectory waits for its own round-i evaluation; other
trajectories keep generating while the device evaluates.

API generation concurrency is adaptive and per provider (AdaptiveLimiter):
- starts at `initial` (3) and doubles (3 -> 6 -> 12 -> 24 -> ...) up to
  `max_limit`, only after real rate-limit headers were received, only when the
  latest headers leave >= 50 % of every reported limit, no 429 and at most one
  overloaded failure happened since the last change, enough requests completed
  at the current level, the host has memory/CPU headroom, and the evaluation
  backlog (candidates waiting for or in evaluation, EvalTracker.pending; not the
  budget reservations of trajectories still waiting to send) is below half its cap;
- an HTTP 429 halves the concurrency and pauses new requests of that provider
  until the provider's Retry-After / reset time;
- generic overloaded / 5xx failures are recorded by the transport protocol;
  only SUSTAINED overload (>= 3 overloaded failures and >= 20 % of the
  provider's attempts in a 10-minute window) halves the concurrency, at most
  once per 10 minutes;
- evaluation backlog, HARD global bound (EvalBudget, shared by every model of
  the process): a generation request may be sent only after its trajectory
  atomically reserved one of `max_pending` slots; the slot is held while the
  request runs, while the candidate waits for and is in evaluation, and is
  released when the round closes. Candidates generated + waiting + being
  evaluated therefore never exceed `max_pending` (revision 3's per-process soft
  cap could overshoot by the in-flight count).
B200 evaluations remain serialized by the device lock (evaluator unchanged).
Graceful pause: creating `<campaign>/STOP` (or `STOP_<model>`) stops every new
request BEFORE anything is sent or logged as sending; in-flight requests
finish and are archived, running evaluations finish, every trajectory pauses
at a resumable boundary and the process exits. Remove the file and run the
same command again to resume.
Every ramp/backoff event and periodic scheduler telemetry (concurrency,
requests/min, tokens/min, 429/overload/unknown-charge rates, RSS/CPU, pending
evaluations) is written under <campaign>/telemetry/; none of it reaches a
prompt or a score."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

from tilebench.llm.v2.orchestration.telemetry import ProcessTelemetry, append_locked
from tilebench.llm.v2.providers.base import ProviderConfigError, StopRequested, TransportError, retry_after_seconds

_OVERLOAD = re.compile(r"overload|\b5\d\d\b|server_error|internal server error|service unavailable|bad gateway", re.I)


def header_headroom(headers: dict | None) -> float | None:
    """min(remaining / limit) over every limit the provider reported; None without pairs."""
    if not headers:
        return None
    fracs = []
    for k, v in headers.items():
        rem_key = None
        if k.startswith("x-ratelimit-limit-"):
            rem_key = "x-ratelimit-remaining-" + k[len("x-ratelimit-limit-"):]
        elif k.startswith(("anthropic-ratelimit-", "anthropic-priority-")) and k.endswith("-limit"):
            rem_key = k[: -len("-limit")] + "-remaining"
        if rem_key and rem_key in headers:
            try:
                lim, rem = float(v), float(headers[rem_key])
            except ValueError:
                continue
            if lim > 0:
                fracs.append(max(0.0, min(1.0, rem / lim)))
    return min(fracs) if fracs else None


def classify_failure(err: BaseException) -> str:
    if isinstance(err, TransportError):
        if err.status_code == 429:
            return "rate_limited"
        if _OVERLOAD.search(f"{err} {err.error_body or ''}") or (err.status_code or 0) >= 500:
            return "overloaded"
    return "error"


class AdaptiveLimiter:
    def __init__(self, provider: str, *, initial: int = 3, max_limit: int = 96, min_limit: int = 1,
                 max_pending: int = 16, pending_fn=lambda: 0, ramp_min_interval_s: float = 120.0,
                 overload_window_s: float = 600.0, events_path: Path | None = None, log=print, clock=time.time,
                 host_ok=None, stop_fn=lambda: False):
        self.provider, self.limit, self.max_limit, self.min_limit = provider, int(initial), int(max_limit), int(min_limit)
        self.max_seen = self.limit
        self.max_pending, self.pending_fn = int(max_pending), pending_fn
        self.ramp_min_interval_s, self.overload_window_s = ramp_min_interval_s, overload_window_s
        self.events_path, self.log, self.clock = events_path, log, clock
        self.host_ok = host_ok or _host_ok
        self.stop_fn = stop_fn
        self.in_flight = 0
        self.pause_until = 0.0
        self.headers_seen = False
        self.last_headers: dict | None = None
        self.last_change = clock()
        self.last_overload_backoff = 0.0
        self.since_change = {"ok": 0, "rate_limited": 0, "overloaded": 0, "error": 0}
        self.history: deque = deque()          # (t, outcome, input_tokens, output_tokens, charged_unknown)
        self.cond = threading.Condition()
        self.closed = False

    # -- gate ------------------------------------------------------------------
    def acquire(self) -> None:
        """A provider slot. The evaluation backlog is bounded separately (EvalBudget, reserved before this call)."""
        with self.cond:
            while True:
                now = self.clock()
                if self.closed or self.stop_fn():
                    raise StopRequested("scheduler pause requested (STOP file); no new request is sent")
                if self.in_flight < self.limit and now >= self.pause_until:
                    self.in_flight += 1
                    return
                self.cond.wait(timeout=1.0)

    def release(self, outcome: str, *, headers: dict | None = None, err: BaseException | None = None,
                usage=None, charged_unknown: bool = False) -> None:
        with self.cond:
            now = self.clock()
            self.in_flight -= 1
            if headers:
                self.last_headers = headers
                if headroom_known(headers):
                    self.headers_seen = True
            it = getattr(usage, "logical_input", None) or 0
            ot = getattr(usage, "logical_output", None) or 0
            self.history.append((now, outcome, it, ot, charged_unknown))
            while self.history and now - self.history[0][0] > 3600:
                self.history.popleft()
            self.since_change[outcome] = self.since_change.get(outcome, 0) + 1
            if outcome == "rate_limited":
                wait = retry_after_seconds(headers or getattr(err, "rate_limit", None))
                self.pause_until = max(self.pause_until, now + max(wait, 1.0))
                self._set(max(self.min_limit, self.limit // 2), "backoff", f"HTTP 429; pause {max(wait, 1.0):.1f}s per the "
                                                                           "provider's Retry-After/reset")
            elif outcome == "overloaded":
                recent = [h for h in self.history if now - h[0] <= self.overload_window_s]
                n_over = sum(1 for h in recent if h[1] == "overloaded")
                if n_over >= 3 and n_over / max(1, len(recent)) >= 0.2 and now - self.last_overload_backoff >= self.overload_window_s:
                    self.last_overload_backoff = now
                    self._set(max(self.min_limit, self.limit // 2), "backoff",
                              f"sustained overload: {n_over}/{len(recent)} attempts overloaded in {self.overload_window_s:.0f}s")
            elif outcome == "ok":
                self._maybe_ramp(now)
            self.cond.notify_all()

    def _maybe_ramp(self, now: float) -> None:
        if self.limit >= self.max_limit or not self.headers_seen:
            return
        if now - self.last_change < self.ramp_min_interval_s or self.since_change.get("ok", 0) < self.limit:
            return
        if self.since_change.get("rate_limited", 0) or self.since_change.get("overloaded", 0) > 1:
            return
        room = header_headroom(self.last_headers)
        if room is None or room < 0.5:
            return
        if self.pending_fn() >= self.max_pending / 2:
            return
        ok, why = self.host_ok()
        if not ok:
            return
        self._set(min(self.max_limit, self.limit * 2), "ramp", f"headroom {room:.2f} >= 0.50, "
                                                               f"{self.since_change.get('ok', 0)} ok since last change, {why}")

    def _set(self, new: int, kind: str, reason: str) -> None:
        old = self.limit
        self.limit = new
        self.max_seen = max(self.max_seen, new)
        self.last_change = self.clock()
        self.since_change = {"ok": 0, "rate_limited": 0, "overloaded": 0, "error": 0}
        ev = {"t": self.last_change, "provider": self.provider, "event": kind, "from": old, "to": new, "reason": reason,
              "pending_evaluations": self.pending_fn(), "in_flight": self.in_flight, "headers": self.last_headers}
        self.log(f"[scheduler:{self.provider}] concurrency {kind} {old} -> {new}: {reason}")
        if self.events_path is not None:
            append_locked(self.events_path, ev)

    # -- reporting ---------------------------------------------------------------
    def stats(self, window_s: float = 300.0) -> dict:
        with self.cond:
            now = self.clock()
            recent = [h for h in self.history if now - h[0] <= window_s]
            n = len(recent)
            minutes = window_s / 60.0
            return {"provider": self.provider, "concurrency_current": self.limit, "concurrency_max_reached": self.max_seen,
                    "concurrency_cap": self.max_limit, "in_flight": self.in_flight,
                    "paused_for_s": max(0.0, self.pause_until - now), "headers_seen": self.headers_seen,
                    "header_headroom": header_headroom(self.last_headers),
                    "requests_per_min": round(n / minutes, 3),
                    "input_tokens_per_min": round(sum(h[2] for h in recent) / minutes, 1),
                    "output_tokens_per_min": round(sum(h[3] for h in recent) / minutes, 1),
                    "rate_429": round(sum(1 for h in recent if h[1] == "rate_limited") / n, 4) if n else 0.0,
                    "rate_overloaded": round(sum(1 for h in recent if h[1] == "overloaded") / n, 4) if n else 0.0,
                    "rate_unknown_charge": round(sum(1 for h in recent if h[4]) / n, 4) if n else 0.0,
                    "window_s": window_s}


def headroom_known(headers: dict | None) -> bool:
    return header_headroom(headers) is not None


def _host_ok() -> tuple[bool, str]:
    try:
        import psutil
        mem = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent(interval=None)
    except Exception:  # noqa: BLE001
        return True, "host metrics unavailable"
    if mem >= 85.0 or cpu >= 90.0:
        return False, f"host busy (mem {mem:.0f}%, cpu {cpu:.0f}%)"
    return True, f"host mem {mem:.0f}%, cpu {cpu:.0f}%"


class EvalBudget:
    """HARD global bound on candidates in the evaluation pipeline: a slot is reserved (atomically, blocking)
    by a trajectory thread before its generation request is sent, kept while the candidate waits for and is
    in evaluation, and released when the round closes (TrajectoryRunner). At most `cap` slots are ever held."""

    def __init__(self, cap: int = 16):
        self.cap = int(cap)
        self.cond = threading.Condition()
        self.used = 0
        self.max_used = 0
        self.reservations = 0
        self._local = threading.local()

    def holds(self) -> bool:
        return bool(getattr(self._local, "held", False))

    def reserve(self, *, stoppable: bool = True, stop_fn=None) -> None:
        if self.holds():
            return
        with self.cond:
            while self.used >= self.cap:
                if stoppable and stop_fn is not None and stop_fn():
                    raise StopRequested("scheduler pause requested (STOP file); no new request is sent")
                self.cond.wait(timeout=1.0)
            if stoppable and stop_fn is not None and stop_fn():
                raise StopRequested("scheduler pause requested (STOP file); no new request is sent")
            self.used += 1
            self.reservations += 1
            self.max_used = max(self.max_used, self.used)
            assert self.used <= self.cap
        self._local.held = True

    def release(self) -> None:
        if not self.holds():
            return
        self._local.held = False
        with self.cond:
            self.used -= 1
            self.cond.notify_all()

    def stats(self) -> dict:
        with self.cond:
            return {"cap": self.cap, "held": self.used, "max_held": self.max_used, "reservations": self.reservations}


class EvalTracker:
    """Wraps the evaluator; counts candidates waiting for or in evaluation (the device lock serializes the GPU)."""

    def __init__(self, inner):
        self.inner = inner
        self.report = inner.report
        self.timeout_s = getattr(inner, "timeout_s", None)
        self._lock = threading.Lock()
        self.pending = 0
        self.max_pending_seen = 0
        self.completed = 0
        self.on_change = None

    def evaluate(self, *a, **kw):
        with self._lock:
            self.pending += 1
            self.max_pending_seen = max(self.max_pending_seen, self.pending)
        try:
            return self.inner.evaluate(*a, **kw)
        finally:
            with self._lock:
                self.pending -= 1
                self.completed += 1
            if self.on_change:
                self.on_change()


class GatedProvider:
    """Provider wrapper: the adaptive gate before every request; outcome, headers and usage fed back after.
    The request and the result pass through unchanged."""

    def __init__(self, inner, limiter: AdaptiveLimiter, telemetry: ProcessTelemetry | None = None,
                 budget: EvalBudget | None = None, stop_fn=None):
        self.inner, self.limiter, self.telemetry = inner, limiter, telemetry
        self.budget = budget
        self.stop_fn = stop_fn or limiter.stop_fn
        self.name = getattr(inner, "name", "provider")
        self.usage_schema = getattr(inner, "usage_schema", None)
        self._local = threading.local()

    def before_send(self) -> None:
        """Called by the runner before it records a `sending` event: reserves an evaluation-backlog slot (hard,
        global), then waits for a provider slot (or raises StopRequested; nothing is sent)."""
        if self.budget is not None:
            self.budget.reserve(stoppable=True, stop_fn=self.stop_fn)
        self.limiter.acquire()
        self._local.slot = True

    def generate(self, request):
        if getattr(self._local, "slot", False):
            self._local.slot = False             # slot taken in before_send
        else:
            self.limiter.acquire()
        if self.telemetry:
            self.telemetry.begin_request()
        try:
            result = self.inner.generate(request)
        except (TransportError, ProviderConfigError) as e:
            self.limiter.release(classify_failure(e), headers=getattr(e, "rate_limit", None), err=e,
                                 charged_unknown=getattr(e, "charged", None) == "unknown")
            if self.telemetry:
                self.telemetry.end_request(ok=False)
            raise
        except BaseException as e:
            self.limiter.release("error", err=e)
            if self.telemetry:
                self.telemetry.end_request(ok=False)
            raise
        self.limiter.release("ok", headers=getattr(result, "rate_limit", None), usage=result.usage,
                             charged_unknown=result.usage.logical_total is None)
        if self.telemetry:
            self.telemetry.end_request(ok=True)
        return result


def run_scheduled(base_spec, dsls: list[str], *, models: list[str] | None = None, initial: int = 3, max_limit: int = 96,
                  max_pending: int = 16, log=print, sleep=time.sleep, provider=None, providers: dict | None = None,
                  evaluator=None, telemetry_interval_s: float = 60.0, review_poll_s: float = 30.0,
                  wait_for_reviews: bool = True) -> dict:
    """Every (model, DSL) track of a campaign in ONE process, trajectories concurrently: per-provider adaptive
    concurrency, one GLOBAL hard evaluation-backlog budget (EvalBudget, `max_pending` slots) and one device
    evaluator. A refused track (preflight blocker) is reported and skipped; the others run. `models` defaults to
    base_spec.model; `providers` ({model: provider}) or `provider` (single model) inject test doubles."""
    from tilebench.llm.v2.evaluation.launcher import SubprocessEvaluator
    from tilebench.llm.v2.manifests import schema as ms
    from tilebench.llm.v2.orchestration.campaign import prepare_track, run_track_task, track_task_dir
    from tilebench.llm.v2.orchestration.identity import LLM_V2_OUTPUT_ROOT
    from tilebench.llm.v2.providers.factory import build_provider, generator_spec

    models_cfg = ms.load_models()
    models = list(models or [base_spec.model])
    out_root = base_spec.out_root or LLM_V2_OUTPUT_ROOT
    cdir = out_root / base_spec.name
    cdir.mkdir(parents=True, exist_ok=True)
    tdir = cdir / "telemetry"
    pid = os.getpid()
    tag = "+".join(models)
    stop_all = cdir / "STOP"
    present = [p.name for p in [stop_all] + [cdir / f"STOP_{m}" for m in models] if p.exists()]
    if present:
        return {"refused": True, "reason": f"pause file(s) {present} present in {cdir}; remove them to resume", "errors": []}
    budget = EvalBudget(max_pending)
    ev = EvalTracker(evaluator or SubprocessEvaluator(device=base_spec.device, timeout_s=base_spec.worker_timeout_s,
                                                       sandbox_root=cdir / "_sandbox", isolation=base_spec.isolation,
                                                       lock_root=out_root))
    telemetry = ProcessTelemetry(tdir / f"scheduler_{base_spec.condition}_{tag}_{pid}.jsonl", interval_s=telemetry_interval_s,
                                 labels={"models": models, "condition": base_spec.condition, "role": "scheduler"})
    limiters: dict[str, AdaptiveLimiter] = {}
    gated: dict[str, GatedProvider] = {}
    for m in models:
        gen = generator_spec(models_cfg, m, accept_status=("approved",) if base_spec.run_type == "formal" else ("approved", "candidate"))
        inner = (providers or {}).get(m) or (provider if len(models) == 1 and provider is not None else None) or \
            build_provider(gen, timeout_s=float(models_cfg.get("transport", {}).get("timeout_s", 3600)))
        stop_m = (lambda m=m: stop_all.exists() or (cdir / f"STOP_{m}").exists())
        lim = AdaptiveLimiter(gen.provider, initial=initial, max_limit=max_limit, max_pending=max_pending,
                              pending_fn=lambda: ev.pending, events_path=tdir / f"concurrency_events_{m}.jsonl",
                              log=log, stop_fn=stop_m)
        limiters[m] = lim
        gated[m] = GatedProvider(inner, lim, telemetry, budget=budget, stop_fn=stop_m)
    ev.on_change = lambda: [_notify(lim) for lim in limiters.values()]
    telemetry.extra.update({"scheduler": lambda: {m: lim.stats() for m, lim in limiters.items()},
                            "eval_budget": lambda: budget.stats(), "pending_evaluations": lambda: ev.pending,
                            "max_pending_evaluations_seen": lambda: ev.max_pending_seen,
                            "evaluations_completed": lambda: ev.completed})
    telemetry.start()
    scheduler_rec = {"kind": "adaptive", "initial": initial, "max_limit": max_limit, "max_pending_evaluations_global": max_pending,
                     "backlog_bound": "hard (EvalBudget reserved before every generation request)",
                     "pid": pid, "dsls": list(dsls), "models": models}
    tracks, refused = [], []
    for m in models:
        for dsl in dsls:
            t = prepare_track(replace(base_spec, dsl=dsl, model=m), log=log, provider=gated[m], evaluator=ev, scheduler=scheduler_rec)
            if isinstance(t, dict):
                refused.append({"dsl": dsl, "model": m, **t})
                log(f"[scheduler] track {m}/{dsl} refused: {t.get('reason') or (t.get('preflight') or {}).get('blockers')}")
            else:
                tracks.append(t)
    jobs = []
    ops = sorted({e.key.operator for t in tracks for e in t.tasks})
    for op in ops:                                      # interleave models and DSLs operator by operator
        for t in tracks:
            jobs.extend((t, e) for e in t.tasks if e.key.operator == op)
    statuses: dict[str, str | None] = {}
    errors: list[dict] = []
    resumed_after_review: list[str] = []
    workers = max(1, len(jobs))
    stopping = lambda: stop_all.exists() or all((cdir / f"STOP_{m}").exists() for m in models)  # noqa: E731
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="traj") as pool:
            futs = {pool.submit(run_track_task, t, e, log=log, sleep=sleep, telemetry=telemetry): (t, e) for t, e in jobs}
            blocked: dict[str, tuple] = {}          # trajectories waiting for a compliance decision (review_required)
            while futs or blocked:
                done, _ = wait(list(futs), timeout=review_poll_s, return_when=FIRST_COMPLETED) if futs else (set(), set())
                for f in done:
                    t, e = futs.pop(f)
                    key = f"{t.spec.model}:{e.key.as_str()}"
                    try:
                        statuses[key] = f.result()
                    except Exception as exc:  # noqa: BLE001 - one trajectory's crash never stops the others
                        errors.append({"task": key, "error": f"{type(exc).__name__}: {exc}"})
                        log(f"[{key}] trajectory thread failed: {type(exc).__name__}: {exc}")
                        continue
                    if statuses[key] == "review_required":
                        blocked[key] = (t, e)
                # the supervising adjudicator resolves reviews out of band (review-resolve); a resolved trajectory
                # continues in THIS process (same provider limiters, same global evaluation budget)
                for key, (t, e) in list(blocked.items()):
                    if stopping():
                        break
                    try:
                        st = json.loads((track_task_dir(t, e) / "trajectory.json").read_text())
                    except (OSError, ValueError):
                        continue
                    if st.get("status") != "review_required":
                        blocked.pop(key)
                        resumed_after_review.append(key)
                        log(f"[{key}] compliance decision recorded; continuing in this process")
                        futs[pool.submit(run_track_task, t, e, log=log, sleep=sleep, telemetry=telemetry)] = (t, e)
                if not futs and blocked and (stopping() or not wait_for_reviews):
                    break
                if not futs and blocked:
                    time.sleep(review_poll_s)
    finally:
        telemetry.stop()
    summary = {"campaign": base_spec.name, "models": models, "scheduler": scheduler_rec,
               "limiter_final": {m: lim.stats() for m, lim in limiters.items()}, "eval_budget": budget.stats(),
               "evaluations_completed": ev.completed, "max_pending_evaluations_seen": ev.max_pending_seen,
               "refused_tracks": refused, "errors": errors, "statuses": statuses, "tracks": [t.summary for t in tracks]}
    summary["resumed_after_review"] = resumed_after_review
    (cdir / f"scheduler_summary_{tag}_{int(time.time())}_{pid}.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return summary


def _notify(limiter: AdaptiveLimiter) -> None:
    with limiter.cond:
        limiter.cond.notify_all()
