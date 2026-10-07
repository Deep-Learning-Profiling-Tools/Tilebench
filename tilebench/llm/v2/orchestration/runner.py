"""Executes one trajectory: render -> request -> parse -> compliance ->
evaluate, persisting every stage, resumable without re-requesting completed
attempts or re-charging their cost.

Persistence per attempt (`round_NN/attempt_M/`, append-only):
    request.json          the exact prompt, model id and settings (hash = request_hash)
    response.json         the archived provider result (written only after a
                          complete response; reused on resume, never re-requested)
    impl_<dsl>.py         the parsed candidate
    compliance.json       the FIRST compliance check: verdict, evidence, the
                          candidate's sha256, the checker fingerprint and the
                          rules sha256 in force; never rewritten (rechecks go
                          to compliance_recheck_NNNN.json)
    eval_NNNN/            one directory per evaluation revision: META.json
                          (reason, supersedes, executor, evaluator fingerprint,
                          candidate sha256), evaluation.json, evaluation/ (the
                          archived sandbox evidence); earlier revisions are
                          never overwritten
Per trajectory:
    trajectory.json       state (atomic replace)
    usage.jsonl           one row per archived response (append-only; reconciled on
                          resume if the process died between response.json and the
                          ledger row)
    transport.jsonl       one row per transport event, the single durable source of
                          an attempt's transport history: sending / succeeded /
                          failed / refused / orphaned / reopened

Transport accounting on (re)start: the attempt's transport list is REBUILT
from every durable event of that (round, attempt), grouped by the monotonic
transport_attempt id: a `sending` without a terminal event becomes a
durable `orphaned` (charge unknown); a `succeeded` whose response was never
archived is a lost response (known charge, counted; a new request follows);
`failed`/`orphaned` with charged=unknown keep propagating. Cost of an
attempt = sum of the known charges; unknown as soon as any charge is
unknown. A prompt that exceeds the context limit is never sent: cost 0,
cost_status not_sent."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from tilebench.llm.v2.evaluation.job import EvaluationJob
from tilebench.llm.v2.orchestration.identity import attempt_dir, request_hash
from tilebench.llm.v2.orchestration.state import AttemptRecord, TrajectoryState
from tilebench.llm.v2.orchestration import state_machine as sm
from tilebench.llm.v2.prompts.renderer import TaskContext, render_initial, render_refinement, render_system
from tilebench.llm.v2.metrics.cost import combine as combine_usd, pricing_sha256, transport_cost
from tilebench.llm.v2.providers.base import (GenerationRequest, GenerationResult, ProviderConfigError, StopRequested,
                                             TransportError, retry_after_seconds)
from tilebench.llm.v2.providers.ledger import append_jsonl, archive_response, read_jsonl
from tilebench.llm.v2.validation.contract_checks import check_compliance
from tilebench.llm.v2.validation.parser import FormatError, parse_single_file

TERMINAL_EVENTS = ("succeeded", "failed", "refused", "orphaned")
# a 429 whose body says the account is out of quota/credit is not retried: the trajectory stops (provider refusal)
_QUOTA = re.compile(r"insufficient_quota|billing_hard_limit|enforced_spend_limit_reached|credit balance|spend limit", re.I)


class Evaluator(Protocol):
    def evaluate(self, source_path: Path, job: EvaluationJob, round_index: int, attempt: int, *,
                 archive_dir: Path | None = None) -> dict: ...


class ConfigMismatch(RuntimeError):
    pass


@dataclass
class RunnerConfig:
    model_id: str
    provider_name: str
    settings: dict
    rounds: int = 5
    max_generations: int = 1               # protocol revision 3: one candidate generation per round
    max_transport_retries: int = 3
    max_rate_limit_retries: int = 12       # HTTP 429 retries (charged "no"), each after the provider's Retry-After/reset
    retry_backoff_s: float = 0.0
    feedback_limits: dict | None = None
    total_prompt_max_chars: int = 260000
    executor: str = "runner"            # recorded in evaluation META / compliance records


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_config_from_source(source: str) -> dict | None:
    """Static read of get_last_config() when it returns a literal dict; else None."""
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_last_config":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                    try:
                        return ast.literal_eval(sub.value)
                    except Exception:
                        return None
    return None


# --------------------------------------------------------------------------
# transport accounting
# --------------------------------------------------------------------------

def attempt_cost(transport: list[dict], usage: dict | None = None) -> tuple[int | None, str]:
    """(cost, cost_status) from the transport records of one attempt.
    `usage` is the normalized usage of the archived final response for
    transport records that carry no usage of their own (legacy archives)."""
    if not transport and usage is None:
        return 0, "not_sent"
    unknown = False
    total = 0
    counted_success = False
    for t in transport:
        oc = t.get("outcome") or t.get("event")
        if oc == "succeeded":
            if "usage" in t and t["usage"] is not None:
                lt = (t["usage"] or {}).get("logical_total")
                if lt is None:
                    unknown = True
                else:
                    total += int(lt)
                    counted_success = True
            elif "usage" in t and t["usage"] is None:
                unknown = True
        elif t.get("charged") == "unknown":
            unknown = True
    if usage is not None and not counted_success:
        lt = usage.get("logical_total")
        if lt is None:
            unknown = True
        else:
            total += int(lt)
    return (None, "unknown") if unknown else (total, "known")


def durable_transport(tdir: Path, round_index: int, attempt: int, *, archived_transport_id: int | None = None) -> list[dict]:
    """Rebuild the transport records of (round, attempt) from transport.jsonl.
    One record per transport_attempt id (monotonic, unique): the last
    terminal event wins; a `sending` without a terminal event is turned
    into a durable `orphaned` event now. `response_archived` marks the
    succeeded record whose response is archived; any other succeeded
    record is a lost response (charge known and counted)."""
    rows = [r for r in read_jsonl(tdir / "transport.jsonl")
            if r.get("round") == round_index and r.get("attempt") == attempt and r.get("transport_attempt") is not None]
    by_id: dict[int, list[dict]] = {}
    for r in rows:
        by_id.setdefault(int(r["transport_attempt"]), []).append(r)
    out: list[dict] = []
    for tid in sorted(by_id):
        evs = by_id[tid]
        terminal = [e for e in evs if e.get("event") in TERMINAL_EVENTS]
        if terminal:
            last = terminal[-1]
            rec = {k: last.get(k) for k in ("round", "attempt", "transport_attempt", "event", "charged", "error", "usage",
                                            "usage_partial", "response_id", "model_id", "terminal_status", "elapsed_s", "t",
                                            "stream_events", "partial_text_chars", "status_code", "kind", "rate_limit",
                                            "last_event", "error_body", "usd")}
            rec["outcome"] = last.get("outcome") or last["event"]
        else:
            rec = {"round": round_index, "attempt": attempt, "transport_attempt": tid, "event": "orphaned", "outcome": "orphaned",
                   "charged": "unknown", "error": "process ended between sending and a terminal event; charge unknown",
                   "t": time.time()}
            append_jsonl(tdir / "transport.jsonl", rec)
        if rec["outcome"] == "succeeded":
            rec["response_archived"] = (archived_transport_id is not None and tid == archived_transport_id)
            if not rec["response_archived"]:
                rec["response_lost"] = True
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def next_revision_name(adir: Path) -> str:
    """Append-only revision numbering of an attempt directory: a legacy
    `evaluation.json` is revision 1, every `eval_NNNN/` keeps its number, the
    next revision is one above the highest existing one (never a re-used
    number, whatever the state file lists)."""
    nums = []
    for p in adir.glob("eval_*"):
        if p.is_dir() and p.name[5:].isdigit():
            nums.append(int(p.name[5:]))
    if (adir / "evaluation.json").exists():
        nums.append(1)
    return f"eval_{max(nums, default=0) + 1:04d}"


class TrajectoryRunner:
    def __init__(self, *, state: TrajectoryState, tdir: Path, ctx: TaskContext, provider, evaluator: Evaluator,
                 job: EvaluationJob, cfg: RunnerConfig, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] | None = None):
        self.state, self.tdir, self.ctx, self.provider, self.evaluator = state, tdir, ctx, provider, evaluator
        self.job, self.cfg, self.sleep = job, cfg, sleep
        self.rules = job.rules
        # hard global evaluation-backlog budget (orchestration.scheduler.EvalBudget) when run under the scheduler:
        # a slot is reserved before a generation request is sent and released when the round closes
        self.budget = getattr(provider, "budget", None)
        self.feedback_limits = cfg.feedback_limits or {"max_diagnostic_lines": 60, "max_diagnostic_chars": 6000}
        self.log = log or (lambda s: None)

    # -- persistence -------------------------------------------------------
    def save(self) -> None:
        self.state.cost_summary = self.state.compute_cost_summary()
        self.state.save(self.tdir / "trajectory.json")

    # -- USD list-price cost (metrics.cost; never part of a prompt) ----------
    def _usd(self, entry: dict, result: GenerationResult | None = None) -> dict:
        raw = result.raw_response if (result is not None and isinstance(result.raw_response, dict)) else {}
        return transport_cost(entry, self.cfg.provider_name, self.cfg.model_id,
                              usage_raw=result.usage_raw if result is not None else None,
                              service_tier=raw.get("service_tier"))

    def _attach_usd(self, transport: list[dict], result: GenerationResult | None = None) -> None:
        for t in transport:
            if "usd" in t and t["usd"] is not None:
                continue
            ok = (t.get("outcome") or t.get("event")) == "succeeded"
            if ok and t.get("response_archived") and result is not None:
                t["usd"] = self._usd(t, result)
            elif ok:
                t["usd"] = {"pricing_snapshot_sha256": pricing_sha256(), "pricing_rule_id": None, "estimated_cost_usd": None,
                            "estimated_cost_lower_bound_usd": None, "usd_cost_status": "unknown", "tokens": None,
                            "note": "succeeded response never archived: raw usage unavailable"}
            else:
                t["usd"] = self._usd(t)

    @staticmethod
    def _usd_fields(transport: list[dict]) -> dict:
        c = combine_usd([t.get("usd") or {"usd_cost_status": "unknown"} for t in transport])
        return {"usd_cost": c["estimated_cost_usd"], "usd_cost_lower_bound": c["estimated_cost_lower_bound_usd"],
                "usd_cost_status": c["usd_cost_status"], "pricing_snapshot_sha256": pricing_sha256()}

    # -- prompt construction -----------------------------------------------
    def _prompt_for(self, action: sm.Action) -> tuple[str, str, str]:
        system = render_system(self.ctx)
        if action.kind == "generate_initial":
            if action.round == 1:
                user = render_initial(self.ctx)
            else:
                prev = self._round_view(action.round - 1)
                best = self._best_view()
                history = [self._round_view(r.round) for r in self.state.rounds if r.valid]
                fallback = None
                if prev.get("source") is None:
                    fb = self.state.last_compliant_source()
                    if fb:
                        fallback = {"round": fb["round"], "source": Path(fb["source_path"]).read_text()}
                user = render_refinement(self.ctx, round_index=action.round, prev=prev, best_valid=best,
                                         history=history, limits=self.feedback_limits, fallback=fallback)
            kind = "initial"
        else:
            raise ValueError(action.kind)
        return system, user, kind

    def _round_view(self, round_index: int) -> dict:
        """What a later prompt may see of a round: its code only when the
        round ended with an evaluated (compliant) candidate; aggregate outcome
        only (valid cases, geometric-mean latency); for an invalid round the
        first failing case's semantic parameters. Never per-case timing."""
        rec = next(r for r in self.state.rounds if r.round == round_index)
        src = None
        if rec.status not in ("contract_violation", "format_error", "review_required"):
            if rec.source_path:
                src = Path(rec.source_path).read_text()
            elif rec.attempts and rec.attempts[-1].verdict in ("clear", "audit_only") and rec.attempts[-1].source_path:
                src = Path(rec.attempts[-1].source_path).read_text()
        ev = rec.evaluation or {}
        first_bad = ev.get("first_failing_case") if rec.status not in ("valid",) else None
        return {"round": rec.round, "status": rec.status, "source": src,
                "configs_distinct": (rec.config or {}).get("configs_distinct") if isinstance(rec.config, dict) else None,
                "latency_ms_geomean": rec.latency_ms_geomean, "valid_cases": rec.valid_cases, "cases_total": rec.cases_total,
                "cases_evaluated": ev.get("cases_evaluated"),
                "first_failing_case": ({"params": first_bad.get("params"), "status": first_bad.get("status")} if first_bad else None),
                "diagnostic": rec.diagnostic}

    def _best_view(self) -> dict | None:
        b = self.state.best_valid
        if not b:
            return None
        return {"round": b["round"], "latency_ms_geomean": b["latency_ms_geomean"],
                "source": Path(b["source_path"]).read_text(), "configs_distinct": b.get("configs_distinct")}

    # -- evaluation backlog budget ------------------------------------------
    def _release_slot(self) -> None:
        if self.budget is not None:
            self.budget.release()

    # -- ledger ------------------------------------------------------------
    def _ledger_has(self, rh: str, round_index: int, attempt: int) -> bool:
        for row in read_jsonl(self.tdir / "usage.jsonl"):
            if row.get("request_hash") == rh and row.get("round") == round_index and row.get("attempt") == attempt:
                return True
        return False

    def _ledger_row(self, action: sm.Action, kind: str, rh: str, result: GenerationResult | None, *,
                    transport: list[dict], reconciled: bool = False, not_sent: bool = False) -> None:
        if not_sent:
            usage = {"status": "not_sent", "logical_total": 0, "logical_input": 0, "logical_output": 0}
        else:
            usage = result.usage.to_dict() if result is not None else None
        usd = self._usd_fields(transport) if transport else {"usd_cost": 0.0, "usd_cost_lower_bound": 0.0,
                                                             "usd_cost_status": "not_sent", "pricing_snapshot_sha256": pricing_sha256()}
        rule_ids = sorted({(t.get("usd") or {}).get("pricing_rule_id") for t in transport} - {None})
        row = {"round": action.round, "attempt": action.attempt, "kind": kind, "request_hash": rh, "usage": usage,
               "model_id": result.model_id if result else None, "response_id": result.response_id if result else None,
               "transport_attempts": len(transport), "transport": transport, "reconciled": reconciled,
               "terminal_status": result.terminal_status if result else None, "t": time.time(),
               "pricing_snapshot_sha256": usd["pricing_snapshot_sha256"], "pricing_rule_id": rule_ids[0] if len(rule_ids) == 1 else rule_ids or None,
               "estimated_cost_usd": usd["usd_cost"], "estimated_cost_lower_bound_usd": usd["usd_cost_lower_bound"],
               "usd_cost_status": usd["usd_cost_status"]}
        append_jsonl(self.tdir / "usage.jsonl", row)

    # -- compliance (first check recorded once; later rechecks are separate files) -----
    def _compliance(self, adir: Path, source: str) -> dict:
        cpath = adir / "compliance.json"
        cand_sha = _sha256_text(source)
        if cpath.exists():
            stored = json.loads(cpath.read_text())
            if stored.get("candidate_sha256") in (None, cand_sha):
                return stored                              # resume: the first verdict stands; no silent re-check
            raise ConfigMismatch(f"{cpath} belongs to a different candidate ({stored.get('candidate_sha256')} != {cand_sha})")
        from tilebench.llm.v2.evaluation.fingerprint import checker_fingerprint
        comp = check_compliance(source, self.ctx.dsl, self.rules)
        comp_d = comp.to_dict()
        comp_d["diagnostics"] = comp.diagnostics()
        comp_d["candidate_sha256"] = cand_sha
        comp_d["checker"] = checker_fingerprint()
        comp_d["rules_sha256"] = self.job.rules_sha256()
        comp_d["checked_at"] = time.time()
        comp_d["executor"] = self.cfg.executor
        cpath.write_text(json.dumps(comp_d, indent=1) + "\n")
        return comp_d

    # -- one generation attempt --------------------------------------------
    def _generate(self, action: sm.Action) -> AttemptRecord:
        system, user, kind = self._prompt_for(action)
        adir = attempt_dir(self.tdir, action.round, action.attempt)
        adir.mkdir(parents=True, exist_ok=True)
        rh = request_hash(system, user, self.cfg.model_id, self.cfg.settings)
        prompt_chars = len(system) + len(user)
        (adir / "request.json").write_text(json.dumps({"request_hash": rh, "system": system, "user": user,
                                                       "model_id": self.cfg.model_id, "settings": self.cfg.settings,
                                                       "provider": self.cfg.provider_name, "kind": kind,
                                                       "prompt_chars": prompt_chars}, indent=1) + "\n")
        if prompt_chars > self.cfg.total_prompt_max_chars:
            if not self._ledger_has(rh, action.round, action.attempt):
                self._ledger_row(action, kind, rh, None, transport=[], not_sent=True)
            return AttemptRecord(attempt=action.attempt, kind=kind, request_hash=rh, prompt_chars=prompt_chars,
                                 transport_attempts=0, verdict="format_error", cost=0, cost_status="not_sent",
                                 usd_cost=0.0, usd_cost_lower_bound=0.0, usd_cost_status="not_sent",
                                 pricing_snapshot_sha256=pricing_sha256(),
                                 diagnostic=f"prompt_too_long: {prompt_chars} chars > {self.cfg.total_prompt_max_chars}; not sent")
        resp_path = adir / "response.json"
        if resp_path.exists():
            # resume: an archived response for this exact request is reused, never re-requested
            archived = json.loads(resp_path.read_text())
            if archived.get("request_hash") != rh:
                raise ConfigMismatch(f"archived response in {adir} was produced by a different request")
            result = GenerationResult.from_dict(archived["result"])
            arch_tid = None
            for t in (archived.get("transport") or []):
                if (t.get("outcome") or t.get("event")) == "succeeded":
                    arch_tid = t.get("transport_attempt")
            transport = durable_transport(self.tdir, action.round, action.attempt, archived_transport_id=arch_tid)
            if not transport:
                transport = list(archived.get("transport") or [])      # legacy archives without durable rows
            self._attach_usd(transport, result)
            if not self._ledger_has(rh, action.round, action.attempt):
                self._ledger_row(action, kind, rh, result, transport=transport, reconciled=True)
                self.state.notes.append(f"round {action.round} attempt {action.attempt}: usage ledger row reconciled from the archived response")
            self.log(f"round {action.round} attempt {action.attempt}: archived response reused (no request)")
        else:
            transport = durable_transport(self.tdir, action.round, action.attempt)
            lost = [t for t in transport if t.get("response_lost")]
            if lost:
                self.state.notes.append(f"round {action.round} attempt {action.attempt}: {len(lost)} succeeded response(s) were never archived; "
                                        "their known charge is kept and a new request follows")
            result, refused = self._request_with_retries(system, user, action, transport)
            self._attach_usd(transport, result)
            if refused is not None:
                cost, cstat = attempt_cost(transport)
                return AttemptRecord(attempt=action.attempt, kind=kind, request_hash=rh, prompt_chars=prompt_chars,
                                     transport_attempts=len(transport), verdict="provider_refused", transport=transport,
                                     cost=cost, cost_status=cstat, error=f"provider refused: {refused}", **self._usd_fields(transport))
            if result is None:
                cost, cstat = attempt_cost(transport)
                return AttemptRecord(attempt=action.attempt, kind=kind, request_hash=rh, prompt_chars=prompt_chars,
                                     transport_attempts=len(transport), verdict="transport_failed", transport=transport,
                                     cost=cost, cost_status=cstat, **self._usd_fields(transport),
                                     error="transport retries exhausted; " + ("charging uncertain" if cstat == "unknown" else "no charge recorded"))
            result.transport_attempts = len(transport)
            archive_response(adir, "response", {"request_hash": rh, "result": result.to_dict(), "transport": transport})
            self._ledger_row(action, kind, rh, result, transport=transport)
        cost, cstat = attempt_cost(transport, result.usage.to_dict())
        base = dict(attempt=action.attempt, kind=kind, request_hash=rh, prompt_chars=prompt_chars,
                    transport_attempts=len(transport), usage=result.usage.to_dict(), cost=cost, cost_status=cstat,
                    **self._usd_fields(transport),
                    transport=transport, response_path=str(resp_path), response_id=result.response_id,
                    model_id=result.model_id, terminal_status=result.terminal_status, truncated=result.truncated)
        if result.truncated:
            return AttemptRecord(verdict="format_error",
                                 diagnostic=f"response truncated by the provider ({result.terminal_status}); not a complete candidate", **base)
        try:
            parsed = parse_single_file(result.text, self.ctx.output_file)
        except FormatError as e:
            return AttemptRecord(verdict="format_error", diagnostic=f"format error: {e}", **base)
        src_path = adir / self.ctx.output_file
        if not src_path.exists():
            src_path.write_text(parsed.source)
        comp_d = self._compliance(adir, parsed.source)
        verdict = {"clear": "clear", "audit_only": "audit_only", "confirmed_violation": "confirmed_violation",
                   "review_required": "review_required"}[comp_d["verdict"]]
        diag = "; ".join(comp_d.get("diagnostics") or []) or \
            ("; ".join(comp_d.get("review_items") or []) if verdict == "review_required" else None)
        return AttemptRecord(verdict=verdict, source_path=str(src_path), config=_read_config_from_source(parsed.source),
                             compliance=comp_d, diagnostic=diag or None, candidate_sha256=comp_d.get("candidate_sha256"),
                             checker_fingerprint=comp_d.get("checker"), rules_sha256=comp_d.get("rules_sha256"), **base)

    def _request_with_retries(self, system: str, user: str, action: sm.Action,
                              transport: list[dict]) -> tuple[GenerationResult | None, str | None]:
        """Returns (result, None) on success, (None, None) when transport retries
        are exhausted in this session, (None, message) when the provider
        refused the request. Transport ids continue from the largest durable
        id of this attempt; the retry budget counts this session's attempts."""
        req = GenerationRequest(system=system, user=user, model_id=self.cfg.model_id, provider=self.cfg.provider_name,
                                settings=self.cfg.settings,
                                metadata={"trajectory": self.state.trajectory_id, "round": action.round, "attempt": action.attempt})
        n = max([int(t["transport_attempt"]) for t in transport if t.get("transport_attempt") is not None], default=0)
        other_failures = rate_limited = 0
        before_send = getattr(self.provider, "before_send", None)
        while True:
            if before_send is not None:
                before_send()          # scheduler gate; raises StopRequested BEFORE anything is sent or logged as sending
            n += 1
            sending = {"round": action.round, "attempt": action.attempt, "transport_attempt": n, "event": "sending",
                       "model_id": self.cfg.model_id, "t": time.time()}
            append_jsonl(self.tdir / "transport.jsonl", sending)
            self.log(f"round {action.round} attempt {action.attempt}: sending request (transport attempt {n})")
            t0 = time.time()
            try:
                result = self.provider.generate(req)
            except ProviderConfigError as e:
                entry = {"round": action.round, "attempt": action.attempt, "transport_attempt": n, "event": "refused",
                         "outcome": "refused", "charged": "no", **e.record(), "elapsed_s": time.time() - t0, "t": time.time()}
                entry["usd"] = self._usd(entry)
                self.state.transport_log.append(entry)
                append_jsonl(self.tdir / "transport.jsonl", entry)
                transport.append(entry)
                self.log(f"round {action.round} attempt {action.attempt}: provider refused: {e}")
                return None, str(e)
            except TransportError as e:
                entry = {"round": action.round, "attempt": action.attempt, "transport_attempt": n, "event": "failed",
                         "outcome": "failed", **e.record(), "elapsed_s": time.time() - t0, "t": time.time()}
                entry["usd"] = self._usd(entry)
                self.state.transport_log.append(entry)
                append_jsonl(self.tdir / "transport.jsonl", entry)
                transport.append(entry)
                self.log(f"round {action.round} attempt {action.attempt}: transport failure ({e.charged} charge): {e}")
                if e.status_code == 429 and _QUOTA.search(f"{e} {e.error_body or ''}"):
                    return None, f"provider quota/billing limit (HTTP 429): {e}"
                if e.status_code == 429:
                    # rate limited before processing (charged "no"): wait for the provider's reset, separate budget
                    rate_limited += 1
                    if rate_limited > self.cfg.max_rate_limit_retries:
                        return None, None
                    self.sleep(max(retry_after_seconds(e.rate_limit), self.cfg.retry_backoff_s))
                    continue
                other_failures += 1
                if other_failures > self.cfg.max_transport_retries:
                    return None, None
                self.sleep(self.cfg.retry_backoff_s * other_failures)
                continue
            entry = {"round": action.round, "attempt": action.attempt, "transport_attempt": n, "event": "succeeded",
                     "outcome": "succeeded", "charged": "known" if result.usage.logical_total is not None else "unknown",
                     "response_id": result.response_id, "model_id": result.model_id,
                     "terminal_status": result.terminal_status, "usage": result.usage.to_dict(),
                     "elapsed_s": time.time() - t0, "t": time.time(), "response_archived": True,
                     "rate_limit": getattr(result, "rate_limit", None)}
            entry["usd"] = self._usd(entry, result)
            append_jsonl(self.tdir / "transport.jsonl", entry)
            transport.append(entry)
            self.log(f"round {action.round} attempt {action.attempt}: response {result.response_id} "
                     f"({result.terminal_status}, {result.usage.logical_total} logical tokens, {entry['elapsed_s']:.0f}s)")
            return result, None

    # -- evaluation revisions (append-only) ---------------------------------
    def _next_revision(self, adir: Path) -> str:
        return next_revision_name(adir)

    # -- main loop ---------------------------------------------------------
    def step(self) -> sm.Action:
        action = sm.next_action(self.state, rounds=self.cfg.rounds, max_generations=self.cfg.max_generations)
        if action.kind in ("done", "blocked_review", "incomplete"):
            self._release_slot()
            return action
        if action.kind == "generate_initial":
            try:
                rec = self._generate(action)
            except StopRequested:
                self._release_slot()
                self.log(f"round {action.round} attempt {action.attempt}: pause requested before sending; nothing sent")
                self.save()
                return sm.Action("paused", round=action.round, attempt=action.attempt)
            except BaseException:
                self._release_slot()
                raise
            sm.apply_attempt(self.state, action.round, rec, max_generations=self.cfg.max_generations, rounds=self.cfg.rounds)
            if rec.verdict not in sm.EVALUABLE_VERDICTS:
                self._release_slot()           # nothing joins the evaluation backlog
            self.log(f"round {action.round} attempt {action.attempt}: verdict {rec.verdict}"
                     + (f" ({rec.diagnostic[:160]})" if rec.diagnostic else ""))
        elif action.kind == "evaluate":
            if self.budget is not None:
                self.budget.reserve(stoppable=False)   # no-op when this round's generation already holds the slot
            try:
                self._evaluate(action)
            finally:
                self._release_slot()
        self.save()
        return action

    def _evaluate(self, action: sm.Action) -> None:
        rec = next(r for r in self.state.rounds if r.round == action.round)
        src = Path(rec.attempts[-1].source_path)
        adir = attempt_dir(self.tdir, action.round, action.attempt)
        rev = self._next_revision(adir)
        rdir = adir / rev
        rdir.mkdir(parents=True, exist_ok=True)
        fp = self.state.evaluator_fingerprint or {}
        meta = {"revision": rev, "reason": rec.evaluation_reason or "initial", "supersedes": rec.evaluation_revision,
                "executor": self.cfg.executor, "evaluator_fingerprint_sha256": fp.get("fingerprint_sha256"),
                "candidate_sha256": _sha256_text(src.read_text()), "worker_timeout_s": getattr(self.evaluator, "timeout_s", None),
                "isolation_backend": (getattr(self.evaluator, "report", None) or {}).get("backend"), "created": time.time()}
        (rdir / "META.json").write_text(json.dumps(meta, indent=1) + "\n")
        result = self.evaluator.evaluate(src, self.job, action.round, action.attempt, archive_dir=rdir / "evaluation")
        result["evaluation_revision"] = rev
        (rdir / "evaluation.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
        sm.apply_evaluation(self.state, action.round, result, max_generations=self.cfg.max_generations, rounds=self.cfg.rounds)
        rec.evaluation_revisions.append(rev)
        rec.evaluation_revision = rev
        rec.evaluation_reason = None
        self.log(f"round {action.round}: evaluation {rec.status} [{rev}] valid cases {rec.valid_cases}/{rec.cases_total}"
                 + (f", geomean {rec.latency_ms_geomean:.4f} ms" if rec.status == "valid" else
                    f" ({str(rec.diagnostic)[:160]})"))
        if self.state.best_valid and self.state.best_valid["round"] == action.round and rec.status == "valid":
            bdir = self.tdir / "best_valid"
            bdir.mkdir(exist_ok=True)
            (bdir / self.ctx.output_file).write_text(src.read_text())

    def closed_rounds(self) -> int:
        return sum(1 for r in self.state.rounds if r.status != "pending")

    def run(self, max_steps: int = 400, stop_after_rounds: int | None = None) -> str:
        """Runs until the trajectory is complete/blocked/incomplete, or until
        `stop_after_rounds` rounds are closed (returns 'paused'; the state on
        disk is resumable)."""
        self.save()
        for _ in range(max_steps):
            if stop_after_rounds is not None and self.closed_rounds() >= stop_after_rounds and \
                    sm.current_round(self.state) is None and self.state.status == "in_progress":
                self.save()
                return "paused"
            action = self.step()
            if action.kind == "paused":
                self.save()
                return "paused"
            if action.kind in ("done", "blocked_review", "incomplete"):
                break
        self.save()
        return self.state.status
