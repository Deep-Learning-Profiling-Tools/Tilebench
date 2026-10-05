"""Evidence access control for distillation.

A distillation run for (dsl, held_out_fold) may read ONLY base-condition
trajectories of that DSL, produced on the DSL's designated source device,
whose operator belongs to one of the two training folds. Both generation
models are pooled. Everything else is refused by code, not by prompt:
- the held-out fold's operators (any dtype, model, device),
- target-device trajectories (GH200/MI300X for the GPU DSLs),
- other DSLs, Enhanced trajectories.
A release-full-data run uses a different, explicitly named mode."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tilebench.llm.v2.manifests.schema import FOLDS, training_folds


class EvidenceAccessError(PermissionError):
    pass


@dataclass(frozen=True)
class TrajectoryRef:
    trajectory_id: str
    dsl: str
    device: str
    operator: str
    dtype: str
    model: str
    condition: str
    fold: str
    path: str


@dataclass
class EvidenceScope:
    dsl: str
    source_device: str
    held_out_fold: str | None            # None only in release mode
    mode: str                            # evaluation | release
    allowed_devices: tuple[str, ...]
    training_folds: tuple[str, ...]
    selected: list[TrajectoryRef] = field(default_factory=list)
    excluded: dict = field(default_factory=dict)

    def allows(self, ref: TrajectoryRef) -> tuple[bool, str]:
        if ref.condition != "base":
            return False, "not a base trajectory"
        if ref.dsl != self.dsl:
            return False, f"other DSL {ref.dsl}"
        if ref.device not in self.allowed_devices:
            return False, f"device {ref.device} is not an allowed source"
        if self.mode == "evaluation" and ref.fold not in self.training_folds:
            return False, f"fold {ref.fold} is held out"
        return True, "ok"

    def open(self, ref: TrajectoryRef) -> Path:
        ok, why = self.allows(ref)
        if not ok or ref not in self.selected:
            raise EvidenceAccessError(f"{ref.trajectory_id}: {why}")
        return Path(ref.path)

    def manifest(self) -> dict:
        return {"dsl": self.dsl, "source_device": self.source_device, "mode": self.mode,
                "held_out_fold": self.held_out_fold, "training_folds": list(self.training_folds),
                "allowed_devices": list(self.allowed_devices),
                "source_trajectory_ids": sorted(r.trajectory_id for r in self.selected),
                "excluded_counts": {k: len(v) for k, v in self.excluded.items()}}


def evaluation_scope(index: list[TrajectoryRef], *, dsl: str, held_out_fold: str, study: dict) -> EvidenceScope:
    if held_out_fold not in FOLDS:
        raise ValueError(held_out_fold)
    src = study["dsls"][dsl]["source_device"]
    scope = EvidenceScope(dsl=dsl, source_device=src, held_out_fold=held_out_fold, mode="evaluation",
                          allowed_devices=(src,), training_folds=training_folds(held_out_fold))
    for ref in index:
        ok, why = scope.allows(ref)
        if ok:
            scope.selected.append(ref)
        else:
            scope.excluded.setdefault(why, []).append(ref.trajectory_id)
    return scope


def release_scope(index: list[TrajectoryRef], *, dsl: str, study: dict) -> EvidenceScope:
    """release-full-data: every base trajectory of the DSL on every device that
    runs it. Separate mode, separate manifest label; never used for evaluation."""
    devices = tuple(study["skill_transfer"][dsl])
    scope = EvidenceScope(dsl=dsl, source_device=study["dsls"][dsl]["source_device"], held_out_fold=None,
                          mode="release", allowed_devices=devices, training_folds=tuple(FOLDS))
    for ref in index:
        ok, why = scope.allows(ref)
        (scope.selected.append(ref) if ok else scope.excluded.setdefault(why, []).append(ref.trajectory_id))
    return scope


def refs_from_states(states: list[dict]) -> list[TrajectoryRef]:
    """Build the index from trajectory.json dicts (+ their paths in 'path')."""
    out = []
    for s in states:
        t = s["task"]
        out.append(TrajectoryRef(trajectory_id=s["trajectory_id"], dsl=t["dsl"], device=t["device"],
                                 operator=t["operator"], dtype=t["dtype"], model=s["model"], condition=s["condition"],
                                 fold=t["fold"], path=s.get("path", "")))
    return out
