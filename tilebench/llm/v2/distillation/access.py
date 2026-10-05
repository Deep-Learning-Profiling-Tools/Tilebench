"""Evidence access control for distillation.

A distillation run for (dsl, held_out_fold) may read ONLY base-condition
trajectories of that DSL, produced on the DSL's designated source device by a
FORMAL campaign, whose operator belongs to one of the two training folds.
Both generation models are pooled. Everything else is refused by code, not
by prompt:
- the held-out fold's operators (any dtype, model, device),
- target-device trajectories (GH200/MI300X for the GPU DSLs),
- other DSLs, Enhanced trajectories,
- validation-run trajectories (engineering acceptance data).
A release-full-data run uses a different, explicitly named mode.

The index is TRUSTED only when built by `build_index` from trajectory.json
files under a campaign root: each reference carries the sha256 of the state
file it was built from, and `EvidenceScope.open` re-verifies that hash,
refuses paths outside the root and refuses symlinked components. The
orchestrator then re-checks the state's own identity fields against the
reference (distillation.orchestrator.verify_state_identity)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from tilebench.llm.v2.manifests.schema import FOLDS, canonical_json, sha256_text, training_folds


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
    run_type: str = "formal"
    state_sha256: str | None = None
    config_hash: str | None = None
    campaign: str | None = None


def _no_symlink(path: Path) -> None:
    cur = path
    for _ in range(len(path.parts)):
        if cur.is_symlink():
            raise EvidenceAccessError(f"{path}: symlinked component {cur}")
        if cur.parent == cur:
            break
        cur = cur.parent


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
    root: str | None = None              # every opened path must resolve under it

    def allows(self, ref: TrajectoryRef) -> tuple[bool, str]:
        if ref.run_type != "formal":
            return False, f"run type {ref.run_type} is not a distillation source"
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
        p = Path(ref.path)
        if self.root is not None:
            root = Path(self.root).resolve()
            try:
                p.resolve().relative_to(root)
            except ValueError:
                raise EvidenceAccessError(f"{ref.trajectory_id}: {p} is outside the evidence root {root}")
        if not p.exists():
            raise EvidenceAccessError(f"{ref.trajectory_id}: {p} does not exist")
        _no_symlink(p)
        if ref.state_sha256 is not None:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h != ref.state_sha256:
                raise EvidenceAccessError(f"{ref.trajectory_id}: state file hash {h[:12]} != index {ref.state_sha256[:12]}")
        return p

    def manifest(self) -> dict:
        return {"dsl": self.dsl, "source_device": self.source_device, "mode": self.mode,
                "held_out_fold": self.held_out_fold, "training_folds": list(self.training_folds),
                "allowed_devices": list(self.allowed_devices), "root": self.root,
                "source_trajectory_ids": sorted(r.trajectory_id for r in self.selected),
                "source_state_sha256": {r.trajectory_id: r.state_sha256 for r in self.selected},
                "excluded_counts": {k: len(v) for k, v in self.excluded.items()}}

    def sha256(self) -> str:
        return sha256_text(canonical_json(self.manifest()))


def evaluation_scope(index: list[TrajectoryRef], *, dsl: str, held_out_fold: str, study: dict,
                     root: Path | str | None = None) -> EvidenceScope:
    if held_out_fold not in FOLDS:
        raise ValueError(held_out_fold)
    src = study["dsls"][dsl]["source_device"]
    scope = EvidenceScope(dsl=dsl, source_device=src, held_out_fold=held_out_fold, mode="evaluation",
                          allowed_devices=(src,), training_folds=training_folds(held_out_fold),
                          root=str(root) if root is not None else None)
    for ref in index:
        ok, why = scope.allows(ref)
        if ok:
            scope.selected.append(ref)
        else:
            scope.excluded.setdefault(why, []).append(ref.trajectory_id)
    return scope


def release_scope(index: list[TrajectoryRef], *, dsl: str, study: dict, root: Path | str | None = None) -> EvidenceScope:
    """release-full-data: every base trajectory of the DSL on every device that
    runs it. Separate mode, separate manifest label; never used for evaluation."""
    devices = tuple(study["skill_transfer"][dsl])
    scope = EvidenceScope(dsl=dsl, source_device=study["dsls"][dsl]["source_device"], held_out_fold=None,
                          mode="release", allowed_devices=devices, training_folds=tuple(FOLDS),
                          root=str(root) if root is not None else None)
    for ref in index:
        ok, why = scope.allows(ref)
        (scope.selected.append(ref) if ok else scope.excluded.setdefault(why, []).append(ref.trajectory_id))
    return scope


def refs_from_states(states: list[dict]) -> list[TrajectoryRef]:
    """Build an (untrusted) index from trajectory.json dicts (+ their paths in
    'path'). Used by tests and by build_index."""
    out = []
    for s in states:
        t = s["task"]
        out.append(TrajectoryRef(trajectory_id=s["trajectory_id"], dsl=t["dsl"], device=t["device"],
                                 operator=t["operator"], dtype=t["dtype"], model=s["model"], condition=s["condition"],
                                 fold=t["fold"], path=s.get("path", ""), run_type=s.get("run_type", "formal"),
                                 state_sha256=s.get("state_sha256"), config_hash=s.get("config_hash"),
                                 campaign=s.get("campaign")))
    return out


def build_index(campaign_root: Path) -> list[TrajectoryRef]:
    """Trusted index: every trajectory.json under the root, with its file hash."""
    root = Path(campaign_root).resolve()
    states = []
    for p in sorted(root.rglob("trajectory.json")):
        _no_symlink(p)
        raw = p.read_bytes()
        s = json.loads(raw)
        s["path"] = str(p)
        s["state_sha256"] = hashlib.sha256(raw).hexdigest()
        states.append(s)
    return refs_from_states(states)
