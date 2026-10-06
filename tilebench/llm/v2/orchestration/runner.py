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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from tilebench.llm.v2.evaluation.job import EvaluationJob
from tilebench.llm.v2.orchestration.identity import attempt_dir, request_hash
from tilebench.llm.v2.orchestration.state import AttemptRecord, TrajectoryState
from tilebench.llm.v2.orchestration import state_machine as sm
from tilebench.llm.v2.prompts.renderer import TaskContext, render_initial, render_refinement, render_repair, render_system
from tilebench.llm.v2.providers.base import GenerationRequest, GenerationResult, ProviderConfigError, TransportError
from tilebench.llm.v2.providers.ledger import append_jsonl, archive_response, read_jsonl
from tilebench.llm.v2.validation.contract_checks import check_compliance
from tilebench.llm.v2.validation.parser import FormatError, parse_single_file

TERMINAL_EVENTS = ("succeeded", "failed", "refused", "orphaned")


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
    rounds: int = 10
    max_generations: int = 3
    max_transport_retries: int = 3
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
                                            "stream_events", "partial_text_chars", "status_code", "kind")}
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
        self.feedback_limits = cfg.feedback_limits or {"max_diagnostic_lines": 60, "max_diagnostic_chars": 6000}
        self.log = log or (lambda s: None)

    # -- persistence -------------------------------------------------------
    def save(self) -> None:
        self.state.save(self.tdir / "trajectory.json")

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
        elif action.kind == "repair":
            rec = next(r for r in self.state.rounds if r.round == action.round)
            rejected = rec.attempts[-1]
            rejected_src = Path(rejected.source_path).read_text() if rejected.source_path else "# (no parsable file)"
            violations = (rejected.compliance or {}).get("diagnostics") or [rejected.diagnostic or "contract violation"]
            fb = self.state.last_compliant_source()
            fallback = None
            if fb:
                fallback = {"round": fb["round"], "source": Path(fb["source_path"]).read_text()}
            user = render_repair(self.ctx, round_index=action.round, attempt=action.attempt,
                                 max_attempts=self.cfg.max_generations, violations=violations,
                                 rejected_source=rejected_src, fallback=fallback)
            kind = "repair"
        else:
            raise ValueError(action.kind)
        return system, user, kind

    def _round_view(self, round_index: int) -> dict:
        """What a later prompt may see of a round: its code only when the
        round ended with an evaluated (compliant) candidate."""
        rec = next(r for r in self.state.rounds if r.round == round_index)
        src = None
        if rec.status not in ("contract_violation", "format_error", "review_required"):
            if rec.source_path:
                src = Path(rec.source_path).read_text()
            elif rec.attempts and rec.attempts[-1].verdict == "clear" and rec.attempts[-1].source_path:
                src = Path(rec.attempts[-1].source_path).read_text()
        return {"round": rec.round, "status": rec.status, "source": src, "config": rec.config,
                "latency_ms_mean": rec.latency_ms_mean, "latency_ms_samples": rec.latency_ms_samples,
                "diagnostic": rec.diagnostic}

    def _best_view(self) -> dict | None:
        b = self.state.best_valid
        if not b:
            return None
        return {"round": b["round"], "latency_ms_mean": b["latency_ms_mean"],
                "source": Path(b["source_path"]).read_text(), "config": b["config"]}

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
        row = {"round": action.round, "attempt": action.attempt, "kind": kind, "request_hash": rh, "usage": usage,
               "model_id": result.model_id if result else None, "response_id": result.response_id if result else None,
               "transport_attempts": len(transport), "transport": transport, "reconciled": reconciled,
               "terminal_status": result.terminal_status if result else None, "t": time.time()}
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
            if refused is not None:
                cost, cstat = attempt_cost(transport)
                return AttemptRecord(attempt=action.attempt, kind=kind, request_hash=rh, prompt_chars=prompt_chars,
                                     transport_attempts=len(transport), verdict="provider_refused", transport=transport,
                                     cost=cost, cost_status=cstat, error=f"provider refused: {refused}")
            if result is None:
                cost, cstat = attempt_cost(transport)
                return AttemptRecord(attempt=action.attempt, kind=kind, request_hash=rh, prompt_chars=prompt_chars,
                                     transport_attempts=len(transport), verdict="transport_failed", transport=transport,
                                     cost=cost, cost_status=cstat,
                                     error="transport retries exhausted; " + ("charging uncertain" if cstat == "unknown" else "no charge recorded"))
            result.transport_attempts = len(transport)
            archive_response(adir, "response", {"request_hash": rh, "result": result.to_dict(), "transport": transport})
            self._ledger_row(action, kind, rh, result, transport=transport)
        cost, cstat = attempt_cost(transport, result.usage.to_dict())
        base = dict(attempt=action.attempt, kind=kind, request_hash=rh, prompt_chars=prompt_chars,
                    transport_attempts=len(transport), usage=result.usage.to_dict(), cost=cost, cost_status=cstat,
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
        verdict = {"clear": "clear", "confirmed_violation": "confirmed_violation",
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
        session_start = n
        while True:
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
                self.state.transport_log.append(entry)
                append_jsonl(self.tdir / "transport.jsonl", entry)
                transport.append(entry)
                self.log(f"round {action.round} attempt {action.attempt}: provider refused: {e}")
                return None, str(e)
            except TransportError as e:
                entry = {"round": action.round, "attempt": action.attempt, "transport_attempt": n, "event": "failed",
                         "outcome": "failed", **e.record(), "elapsed_s": time.time() - t0, "t": time.time()}
                self.state.transport_log.append(entry)
                append_jsonl(self.tdir / "transport.jsonl", entry)
                transport.append(entry)
                self.log(f"round {action.round} attempt {action.attempt}: transport failure ({e.charged} charge): {e}")
                if n - session_start > self.cfg.max_transport_retries:
                    return None, None
                self.sleep(self.cfg.retry_backoff_s * (n - session_start))
                continue
            entry = {"round": action.round, "attempt": action.attempt, "transport_attempt": n, "event": "succeeded",
                     "outcome": "succeeded", "charged": "known" if result.usage.logical_total is not None else "unknown",
                     "response_id": result.response_id, "model_id": result.model_id,
                     "terminal_status": result.terminal_status, "usage": result.usage.to_dict(),
                     "elapsed_s": time.time() - t0, "t": time.time(), "response_archived": True}
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
            return action
        if action.kind in ("generate_initial", "repair"):
            rec = self._generate(action)
            sm.apply_attempt(self.state, action.round, rec, max_generations=self.cfg.max_generations)
            self.log(f"round {action.round} attempt {action.attempt}: verdict {rec.verdict}"
                     + (f" ({rec.diagnostic[:160]})" if rec.diagnostic else ""))
        elif action.kind == "evaluate":
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
            sm.apply_evaluation(self.state, action.round, result, max_generations=self.cfg.max_generations)
            rec.evaluation_revisions.append(rev)
            rec.evaluation_revision = rev
            rec.evaluation_reason = None
            self.log(f"round {action.round}: evaluation {result.get('status')} [{rev}]"
                     + (f" {result.get('latency_ms_mean'):.4f} ms {result.get('latency_ms_samples')}" if result.get("status") == "valid" else
                        f" ({str(result.get('diagnostic'))[:160]})"))
            if self.state.best_valid and self.state.best_valid["round"] == action.round and rec.status == "valid":
                bdir = self.tdir / "best_valid"
                bdir.mkdir(exist_ok=True)
                (bdir / self.ctx.output_file).write_text(src.read_text())
        self.save()
        return action

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
            if action.kind in ("done", "blocked_review", "incomplete"):
                break
        self.save()
        return self.state.status
