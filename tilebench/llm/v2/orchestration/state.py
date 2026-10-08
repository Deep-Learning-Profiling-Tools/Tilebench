"""Trajectory state records (JSON-serializable dataclasses)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# /4: protocol revision 4 (all 20 cases of the selected dtype per candidate, 5 rounds, one generation per round,
# checker v2, Claude Code review). /3 (revision 3) and /1, /2 (revision 2) are pilot states, still readable.
SCHEMA = "tilebench-llm-v2-trajectory/4"

ATTEMPT_VERDICTS = ("clear", "audit_only", "confirmed_violation", "review_required", "format_error", "transport_failed",
                    "provider_refused")
ROUND_STATUSES = ("pending", "valid", "format_error", "interface_error", "compile_error", "runtime_error",
                  "numerical_error", "timing_error", "contract_violation", "review_required", "infrastructure_incomplete")
TRAJECTORY_STATUSES = ("in_progress", "complete", "incomplete", "review_required")
# cost_status of an attempt:
#   known         every transport attempt's charge is known; cost is exact
#   not_sent      no request reached a provider (e.g. prompt_too_long); cost 0
#   unknown       at least one transport attempt may have been billed an unknown amount
COST_STATUSES = ("known", "not_sent", "unknown")


@dataclass
class AttemptRecord:
    attempt: int                          # always 1 under revision 3 (revision-2 pilot states: 1..3)
    kind: str                             # initial (revision-2 pilot states may hold `repair`)
    request_hash: str
    prompt_chars: int
    transport_attempts: int
    verdict: str                          # one of ATTEMPT_VERDICTS
    usage: dict | None = None             # NormalizedUsage.to_dict() of the successful response
    cost: int | None = None               # logical tokens charged for this attempt (all transport attempts), None = unknown
    cost_status: str = "known"            # one of COST_STATUSES
    transport: list[dict] = field(default_factory=list)   # per transport attempt: {transport_attempt, outcome, charged, usage/error}
    response_path: str | None = None
    source_path: str | None = None
    config: dict | None = None
    compliance: dict | None = None
    diagnostic: str | None = None
    error: str | None = None
    response_id: str | None = None
    model_id: str | None = None           # echoed by the provider
    terminal_status: str | None = None
    truncated: bool = False
    execution: dict | None = None         # execution-side evidence when the worker confirmed a violation
    candidate_sha256: str | None = None   # of the parsed candidate file
    checker_fingerprint: dict | None = None   # checker version + sources sha at the FIRST compliance check
    rules_sha256: str | None = None       # evaluator rules applied at the first compliance check
    compliance_revisions: list[str] = field(default_factory=list)   # append-only recheck files
    # USD list-price cost of all transport attempts of this generation (metrics.cost); None = not exact
    usd_cost: float | None = None
    usd_cost_lower_bound: float | None = None
    usd_cost_status: str = "unknown"      # one of metrics.cost.USD_STATUSES
    pricing_snapshot_sha256: str | None = None


@dataclass
class RoundRecord:
    round: int
    attempts: list[AttemptRecord] = field(default_factory=list)
    status: str = "pending"
    evaluation: dict | None = None        # worker result (stage records, timing)
    latency_ms_mean: float | None = None
    latency_ms_samples: list[float] | None = None
    source_path: str | None = None
    config: dict | None = None
    diagnostic: str | None = None
    timing_execution_mode: str | None = None
    timing_mode_differs: bool | None = None
    evaluation_revision: str | None = None    # eval_NNNN directory of the evaluation the round status comes from
    evaluation_revisions: list[str] = field(default_factory=list)   # every evaluation revision ever produced (append-only)
    evaluation_reason: str | None = None      # why the next evaluation runs (None = initial; set by retry_incomplete)
    reopened_attempts: list[dict] = field(default_factory=list)     # attempt records removed by an explicit transport reopen
    # revision 4 (case suite): round latency = geometric mean of the case means; valid only with every case valid
    latency_ms_geomean: float | None = None
    valid_cases: int | None = None
    cases_total: int | None = None
    case_results: list[dict] = field(default_factory=list)   # compact per case: id, status, mean, 3 samples, config, mode
    reviews: list[dict] = field(default_factory=list)        # compliance adjudications of this round (append-only)

    @property
    def valid(self) -> bool:
        return self.status == "valid"


@dataclass
class TrajectoryState:
    schema: str
    trajectory_id: str
    task: dict                            # operator, dtype, case_id, params, device, dsl, fold, fp8_format
    model: str
    condition: str
    config_hash: str                      # study/models/folds/modes hash
    content_hashes: dict                  # skill components + contract + templates
    output_file: str
    rounds: list[RoundRecord] = field(default_factory=list)
    best_valid: dict | None = None        # {round, latency_ms_geomean, source_path, configs}
    status: str = "in_progress"
    transport_log: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    run_type: str = "formal"              # formal | validation
    campaign: str | None = None
    generator: dict | None = None         # GeneratorSpec.record() (no secrets)
    evaluation_job: dict | None = None    # EvaluationJob.record()
    stop_reason: str | None = None        # why status is incomplete (provider refusal, transport exhaustion, ...)
    evaluator_fingerprint: dict | None = None   # evaluation.fingerprint record at trajectory creation
    evaluator_changes: list[dict] = field(default_factory=list)   # recorded (validation-only) evaluator changes on resume
    scoring_binding: dict | None = None   # metrics.empirical.scoring_binding at creation: this device's profile sha + declaration sha
    protocol: dict | None = None          # {revision, rounds, max_generations_per_round, task_unit, checker_version} at creation
    pricing_binding: dict | None = None   # metrics.cost.pricing_binding() at creation (formal resume refuses a change)
    cost_summary: dict | None = None      # derived on every save: per-round cumulative tokens / USD, trajectory totals

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryState":
        rounds = [RoundRecord(**{**r, "attempts": [AttemptRecord(**a) for a in r.get("attempts", [])]})
                  for r in d.get("rounds", [])]
        return cls(**{**d, "rounds": rounds})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "TrajectoryState":
        return cls.from_dict(json.loads(path.read_text()))

    # -- views -------------------------------------------------------------
    def valid_rounds(self) -> list[RoundRecord]:
        return [r for r in self.rounds if r.valid]

    def metric_rounds(self) -> list[dict]:
        """Rounds in the shape metrics.efficiency expects (token cost and, separately, USD cost per attempt).
        Case-suite rounds carry `case_latency_ms` {case_id: mean of 3} and their round latency is the geometric
        mean over the cases; single-case (pilot) rounds keep latency_ms / latency_ms_samples."""
        out = []
        for r in self.rounds:
            row = {"round": r.round, "attempts": [{"cost": a.cost, "cost_status": a.cost_status, "usd": a.usd_cost,
                                                   "usd_status": a.usd_cost_status} for a in r.attempts],
                   "valid": r.valid, "timing_execution_mode": r.timing_execution_mode, "timing_mode_differs": r.timing_mode_differs}
            if r.cases_total:
                row.update({"latency_ms": r.latency_ms_geomean, "valid_cases": r.valid_cases, "cases_total": r.cases_total,
                            "case_latency_ms": {c["case_id"]: c.get("latency_ms_mean") for c in r.case_results
                                                if c.get("status") == "valid"},
                            "case_samples_ms": {c["case_id"]: c.get("latency_ms_samples") for c in r.case_results
                                                if c.get("status") == "valid"}})
            else:
                row.update({"latency_ms": r.latency_ms_mean, "latency_ms_samples": r.latency_ms_samples})
            out.append(row)
        return out

    def last_compliant_source(self) -> dict | None:
        """The most recent attempt that cleared compliance and was not later
        found violating at execution (fallback for repair/refinement prompts)."""
        for r in reversed(self.rounds):
            for a in reversed(r.attempts):
                if a.verdict in ("clear", "audit_only") and a.source_path:
                    return {"round": r.round, "source_path": a.source_path, "config": a.config}
        return None

    def request_count(self) -> int:
        return sum(len(a.transport) for r in self.rounds for a in r.attempts)

    def compute_cost_summary(self) -> dict:
        """Per-round cumulative logical tokens and estimated USD (None from the first unknown on), and
        trajectory totals. Tokens and USD are separate axes."""
        rows, tok, usd, usd_lb = [], 0, 0.0, 0.0
        tok_known, usd_known = True, True
        unknown_attempts = 0
        for r in self.rounds:
            r_tok = r_usd = 0
            for a in r.attempts:
                if a.cost is None:
                    tok_known = False
                else:
                    tok += int(a.cost)
                    r_tok += int(a.cost)
                if a.usd_cost is None:
                    usd_known = False
                    unknown_attempts += 1
                    usd_lb += float(a.usd_cost_lower_bound or 0.0)
                else:
                    usd += float(a.usd_cost)
                    usd_lb += float(a.usd_cost)
                    r_usd += float(a.usd_cost)
            rows.append({"round": r.round, "logical_tokens": r_tok, "estimated_usd_known": round(r_usd, 10),
                         "cumulative_logical_tokens": tok if tok_known else None,
                         "cumulative_estimated_usd": round(usd, 10) if usd_known else None,
                         "cumulative_estimated_usd_lower_bound": round(usd_lb, 10)})
        return {"rounds": rows, "known_logical_tokens": tok, "logical_tokens_exact": tok_known,
                "known_estimated_usd": round(usd, 10), "estimated_usd_lower_bound": round(usd_lb, 10),
                "usd_cost_exact": usd_known, "unknown_cost_attempts": unknown_attempts,
                "pricing_snapshot_sha256": (self.pricing_binding or {}).get("pricing_snapshot_sha256")}
