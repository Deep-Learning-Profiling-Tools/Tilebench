"""Trajectory state records (JSON-serializable dataclasses)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA = "tilebench-llm-v2-trajectory/1"

ATTEMPT_VERDICTS = ("clear", "confirmed_violation", "review_required", "format_error", "transport_failed")
ROUND_STATUSES = ("pending", "valid", "format_error", "compile_error", "runtime_error", "numerical_error",
                  "timing_error", "contract_violation", "review_required", "infrastructure_incomplete")
TRAJECTORY_STATUSES = ("in_progress", "complete", "incomplete", "review_required")


@dataclass
class AttemptRecord:
    attempt: int                          # 1..3
    kind: str                             # initial | repair
    request_hash: str
    prompt_chars: int
    transport_attempts: int
    verdict: str                          # one of ATTEMPT_VERDICTS
    usage: dict | None = None             # NormalizedUsage.to_dict()
    cost: int | None = None               # logical_total or None when unknown
    response_path: str | None = None
    source_path: str | None = None
    config: dict | None = None
    compliance: dict | None = None
    diagnostic: str | None = None
    error: str | None = None


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
    best_valid: dict | None = None        # {round, latency_ms_mean, source_path, config}
    status: str = "in_progress"
    transport_log: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

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
        """Rounds in the shape metrics.efficiency expects."""
        return [{"round": r.round, "attempts": [{"cost": a.cost} for a in r.attempts],
                 "valid": r.valid, "latency_ms": r.latency_ms_mean} for r in self.rounds]

    def last_compliant_source(self) -> dict | None:
        """The most recent attempt that cleared compliance (fallback for repair prompts)."""
        for r in reversed(self.rounds):
            for a in reversed(r.attempts):
                if a.verdict == "clear" and a.source_path:
                    return {"round": r.round, "source_path": a.source_path, "config": a.config}
        return None
