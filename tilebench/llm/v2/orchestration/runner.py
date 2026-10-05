"""Executes one trajectory: render -> request -> parse -> compliance ->
evaluate, persisting every stage, resumable without re-requesting completed
attempts or re-charging their cost."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from tilebench.llm.v2.orchestration.identity import attempt_dir, request_hash
from tilebench.llm.v2.orchestration.state import AttemptRecord, TrajectoryState
from tilebench.llm.v2.orchestration import state_machine as sm
from tilebench.llm.v2.prompts.renderer import TaskContext, render_initial, render_refinement, render_repair, render_system
from tilebench.llm.v2.providers.base import GenerationRequest, GenerationResult, TransportError
from tilebench.llm.v2.providers.ledger import append_jsonl, archive_response
from tilebench.llm.v2.validation.contract_checks import check_compliance
from tilebench.llm.v2.validation.parser import FormatError, parse_single_file


class Evaluator(Protocol):
    def evaluate(self, source_path: Path, task: dict, round_index: int, attempt: int) -> dict: ...


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


class TrajectoryRunner:
    def __init__(self, *, state: TrajectoryState, tdir: Path, ctx: TaskContext, provider, evaluator: Evaluator,
                 rules: dict, cfg: RunnerConfig, sleep: Callable[[float], None] = time.sleep):
        self.state, self.tdir, self.ctx, self.provider, self.evaluator = state, tdir, ctx, provider, evaluator
        self.rules, self.cfg, self.sleep = rules, cfg, sleep
        self.feedback_limits = cfg.feedback_limits or {"max_diagnostic_lines": 60, "max_diagnostic_chars": 6000}

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
                user = render_refinement(self.ctx, round_index=action.round, prev=prev, best_valid=best,
                                         history=history, limits=self.feedback_limits)
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
        rec = next(r for r in self.state.rounds if r.round == round_index)
        src = Path(rec.source_path).read_text() if rec.source_path else None
        if src is None and rec.attempts and rec.attempts[-1].source_path:
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

    # -- one generation attempt --------------------------------------------
    def _generate(self, action: sm.Action) -> AttemptRecord:
        system, user, kind = self._prompt_for(action)
        adir = attempt_dir(self.tdir, action.round, action.attempt)
        adir.mkdir(parents=True, exist_ok=True)
        rh = request_hash(system, user, self.cfg.model_id, self.cfg.settings)
        prompt_chars = len(system) + len(user)
        (adir / "request.json").write_text(json.dumps({"request_hash": rh, "system": system, "user": user,
                                                       "model_id": self.cfg.model_id, "settings": self.cfg.settings,
                                                       "kind": kind}, indent=1) + "\n")
        if prompt_chars > self.cfg.total_prompt_max_chars:
            return AttemptRecord(attempt=action.attempt, kind=kind, request_hash=rh, prompt_chars=prompt_chars,
                                 transport_attempts=0, verdict="format_error",
                                 diagnostic=f"prompt_too_long: {prompt_chars} chars > {self.cfg.total_prompt_max_chars}")
        # resume: an archived response for this exact request is reused, never re-requested
        resp_path = adir / "response.json"
        if resp_path.exists():
            archived = json.loads(resp_path.read_text())
            if archived.get("request_hash") != rh:
                raise ConfigMismatch(f"archived response in {adir} was produced by a different request")
            result = GenerationResult(**{k: v for k, v in archived["result"].items() if k != "usage"},
                                      usage=_usage_from_dict(archived["result"]["usage"]))
            transport_attempts = archived["result"]["transport_attempts"]
        else:
            result, transport_attempts = self._request_with_retries(system, user, action)
            if result is None:
                return AttemptRecord(attempt=action.attempt, kind=kind, request_hash=rh, prompt_chars=prompt_chars,
                                     transport_attempts=transport_attempts, verdict="transport_failed",
                                     error="transport retries exhausted; usage unknown, charging uncertain")
            result.transport_attempts = transport_attempts
            archive_response(adir, "response", {"request_hash": rh, "result": result.to_dict()})
            append_jsonl(self.tdir / "usage.jsonl", {"round": action.round, "attempt": action.attempt, "kind": kind,
                                                     "request_hash": rh, "usage": result.usage.to_dict(),
                                                     "model_id": result.model_id, "response_id": result.response_id,
                                                     "transport_attempts": transport_attempts})
        cost = result.usage.logical_total
        base = dict(attempt=action.attempt, kind=kind, request_hash=rh, prompt_chars=prompt_chars,
                    transport_attempts=transport_attempts, usage=result.usage.to_dict(), cost=cost,
                    response_path=str(resp_path))
        try:
            parsed = parse_single_file(result.text, self.ctx.output_file)
        except FormatError as e:
            return AttemptRecord(verdict="format_error", diagnostic=f"format error: {e}", **base)
        src_path = adir / self.ctx.output_file
        src_path.write_text(parsed.source)
        comp = check_compliance(parsed.source, self.ctx.dsl, self.rules)
        comp_d = comp.to_dict()
        comp_d["diagnostics"] = comp.diagnostics()
        (adir / "compliance.json").write_text(json.dumps(comp_d, indent=1) + "\n")
        verdict = {"clear": "clear", "confirmed_violation": "confirmed_violation",
                   "review_required": "review_required"}[comp.verdict]
        return AttemptRecord(verdict=verdict, source_path=str(src_path), config=_read_config_from_source(parsed.source),
                             compliance=comp_d, diagnostic="; ".join(comp.diagnostics()) or None, **base)

    def _request_with_retries(self, system: str, user: str, action: sm.Action) -> tuple[GenerationResult | None, int]:
        req = GenerationRequest(system=system, user=user, model_id=self.cfg.model_id, provider=self.cfg.provider_name,
                                settings=self.cfg.settings,
                                metadata={"trajectory": self.state.trajectory_id, "round": action.round, "attempt": action.attempt})
        attempts = 0
        while True:
            attempts += 1
            try:
                return self.provider.generate(req), attempts
            except TransportError as e:
                entry = {"round": action.round, "attempt": action.attempt, "transport_attempt": attempts, "error": str(e)}
                self.state.transport_log.append(entry)
                append_jsonl(self.tdir / "transport.jsonl", entry)
                if attempts > self.cfg.max_transport_retries:
                    return None, attempts
                self.sleep(self.cfg.retry_backoff_s * attempts)

    # -- main loop ---------------------------------------------------------
    def step(self) -> sm.Action:
        action = sm.next_action(self.state, rounds=self.cfg.rounds, max_generations=self.cfg.max_generations)
        if action.kind in ("done", "blocked_review", "incomplete"):
            return action
        if action.kind in ("generate_initial", "repair"):
            rec = self._generate(action)
            sm.apply_attempt(self.state, action.round, rec, max_generations=self.cfg.max_generations)
        elif action.kind == "evaluate":
            rec = next(r for r in self.state.rounds if r.round == action.round)
            src = Path(rec.attempts[-1].source_path)
            result = self.evaluator.evaluate(src, self.state.task, action.round, action.attempt)
            (self.tdir / f"round_{action.round:02d}" / "evaluation.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
            sm.apply_evaluation(self.state, action.round, result)
            if self.state.best_valid and self.state.best_valid["round"] == action.round:
                bdir = self.tdir / "best_valid"
                bdir.mkdir(exist_ok=True)
                (bdir / self.ctx.output_file).write_text(src.read_text())
        self.save()
        return action

    def run(self, max_steps: int = 200) -> str:
        for _ in range(max_steps):
            action = self.step()
            if action.kind in ("done", "blocked_review", "incomplete"):
                break
        self.save()
        return self.state.status


def _usage_from_dict(d: dict):
    from tilebench.llm.v2.providers.usage import NormalizedUsage
    return NormalizedUsage(**d)
