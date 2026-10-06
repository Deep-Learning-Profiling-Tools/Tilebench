"""Evidence access control for distillation.

A distillation run for (dsl, held_out_fold) may read ONLY complete,
base-condition trajectories of that DSL, produced on the DSL's designated
source device by a FORMAL campaign, whose operator belongs to one of the two
training folds OF THE FROZEN FOLD MANIFEST. Both generation models are
pooled. Everything else is refused by code, not by prompt:
- the held-out fold's operators (any dtype, model, device),
- target-device trajectories (GH200/MI300X for the GPU DSLs),
- other DSLs, Enhanced trajectories,
- validation-run trajectories (engineering acceptance data),
- incomplete / blocked trajectories (missing rounds are never silently
  accepted; a complete trajectory with failed rounds is legitimate),
- states whose own fold label disagrees with the frozen manifest.
A release-full-data run uses a different, explicitly named mode.

The index is TRUSTED only when built by `build_index` from trajectory.json
files under a campaign root with the frozen fold manifest: each reference
carries the sha256 of the state file it was built from, its fold is
RECOMPUTED from the manifest (the state's label is kept for comparison
only), and `EvidenceScope.open` re-verifies hash, root containment and
symlinks. The orchestrator then re-checks the state's identity against the
reference and the manifest (distillation.orchestrator.verify_state_identity).

Coverage: `coverage_report` compares the selected references with the
pre-declared task set of the source campaign (training-fold operators x
their dtypes x the campaign's models); formal distillation refuses on
missing or incomplete coverage unless the caller explicitly allows a
partial source and records it."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from tilebench.llm.v2.manifests.schema import FOLDS, ManifestError, canonical_json, fold_of, sha256_text, training_folds

ACCEPTED_STATE_SCHEMAS = ("tilebench-llm-v2-trajectory/1", "tilebench-llm-v2-trajectory/2")


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
    fold: str                            # recomputed from the frozen fold manifest when one is given
    path: str
    run_type: str = "formal"
    state_sha256: str | None = None
    config_hash: str | None = None
    campaign: str | None = None
    status: str = "complete"
    schema: str | None = None
    state_fold: str | None = None        # the state's own label (for the mismatch check)
    rounds: int = 0


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
    folds_status: str | None = None      # status of the fold manifest used (frozen | proposed)

    def allows(self, ref: TrajectoryRef) -> tuple[bool, str]:
        if ref.run_type != "formal":
            return False, f"run type {ref.run_type} is not a distillation source"
        if ref.schema is not None and ref.schema not in ACCEPTED_STATE_SCHEMAS:
            return False, f"state schema {ref.schema} not accepted"
        if ref.condition != "base":
            return False, "not a base trajectory"
        if ref.dsl != self.dsl:
            return False, f"other DSL {ref.dsl}"
        if ref.device not in self.allowed_devices:
            return False, f"device {ref.device} is not an allowed source"
        if ref.state_fold is not None and ref.state_fold != ref.fold:
            return False, f"state fold label {ref.state_fold} disagrees with the frozen fold manifest ({ref.fold})"
        if self.mode == "evaluation" and ref.fold not in self.training_folds:
            return False, f"fold {ref.fold} is held out"
        if ref.status != "complete":
            return False, f"trajectory not complete (status {ref.status}, {ref.rounds} rounds)"
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
                "allowed_devices": list(self.allowed_devices), "root": self.root, "folds_status": self.folds_status,
                "source_trajectory_ids": sorted(r.trajectory_id for r in self.selected),
                "source_state_sha256": {r.trajectory_id: r.state_sha256 for r in self.selected},
                "source_campaigns": sorted({r.campaign for r in self.selected if r.campaign}),
                "excluded_counts": {k: len(v) for k, v in self.excluded.items()}}

    def sha256(self) -> str:
        return sha256_text(canonical_json(self.manifest()))


def _scope(index: list[TrajectoryRef], scope: EvidenceScope) -> EvidenceScope:
    for ref in index:
        ok, why = scope.allows(ref)
        if ok:
            scope.selected.append(ref)
        else:
            scope.excluded.setdefault(why, []).append(ref.trajectory_id)
    return scope


def evaluation_scope(index: list[TrajectoryRef], *, dsl: str, held_out_fold: str, study: dict,
                     root: Path | str | None = None, folds: dict | None = None) -> EvidenceScope:
    if held_out_fold not in FOLDS:
        raise ValueError(held_out_fold)
    src = study["dsls"][dsl]["source_device"]
    scope = EvidenceScope(dsl=dsl, source_device=src, held_out_fold=held_out_fold, mode="evaluation",
                          allowed_devices=(src,), training_folds=training_folds(held_out_fold),
                          root=str(root) if root is not None else None,
                          folds_status=(folds or {}).get("status"))
    return _scope(index, scope)


def release_scope(index: list[TrajectoryRef], *, dsl: str, study: dict, root: Path | str | None = None,
                  folds: dict | None = None) -> EvidenceScope:
    """release-full-data: every base trajectory of the DSL on every device that
    runs it. Separate mode, separate manifest label; never used for evaluation."""
    devices = tuple(study["skill_transfer"][dsl])
    scope = EvidenceScope(dsl=dsl, source_device=study["dsls"][dsl]["source_device"], held_out_fold=None,
                          mode="release", allowed_devices=devices, training_folds=tuple(FOLDS),
                          root=str(root) if root is not None else None, folds_status=(folds or {}).get("status"))
    return _scope(index, scope)


def refs_from_states(states: list[dict], folds: dict | None = None) -> list[TrajectoryRef]:
    """Build the index from trajectory.json dicts (+ their paths in 'path').
    With `folds`, the fold is RECOMPUTED from the manifest and the state's
    own label is kept in `state_fold`; without it (tests, legacy) the state's
    label is used and no mismatch can be detected."""
    out = []
    for s in states:
        t = s["task"]
        state_fold = t.get("fold")
        if folds is not None:
            try:
                fold = fold_of(folds, t["operator"])
            except ManifestError:
                fold = "?"
        else:
            fold = state_fold
        out.append(TrajectoryRef(trajectory_id=s["trajectory_id"], dsl=t["dsl"], device=t["device"],
                                 operator=t["operator"], dtype=t["dtype"], model=s["model"], condition=s["condition"],
                                 fold=fold, path=s.get("path", ""), run_type=s.get("run_type", "formal"),
                                 state_sha256=s.get("state_sha256"), config_hash=s.get("config_hash"),
                                 campaign=s.get("campaign"), status=s.get("status", "complete"), schema=s.get("schema"),
                                 state_fold=state_fold if folds is not None else None,
                                 rounds=sum(1 for r in s.get("rounds", []) if r.get("status") != "pending")))
    return out


def build_index(campaign_root: Path, *, folds: dict | None = None) -> list[TrajectoryRef]:
    """Trusted index: every trajectory.json under the root, with its file hash
    and (with `folds`) the fold recomputed from the frozen manifest."""
    root = Path(campaign_root).resolve()
    states = []
    for p in sorted(root.rglob("trajectory.json")):
        if "_sandbox" in p.parts:
            continue
        _no_symlink(p)
        raw = p.read_bytes()
        s = json.loads(raw)
        s["path"] = str(p)
        s["state_sha256"] = hashlib.sha256(raw).hexdigest()
        states.append(s)
    return refs_from_states(states, folds)


def coverage_report(scope: EvidenceScope, *, study: dict, folds: dict, models: list[str],
                    expected_tasks: list[tuple[str, str]] | None = None) -> dict:
    """Expected (operator, dtype, model) of the training folds on the source
    device versus what the scope selected. `expected_tasks` (operator,
    dtype) pairs default to every eligible task of the source device/DSL
    whose operator is in a training fold (tasks.support.task_table)."""
    if expected_tasks is None:
        from tilebench.llm.v2.tasks.support import task_table
        expected_tasks = [(e.key.operator, e.key.dtype) for e in task_table(study, folds)
                          if e.key.device == scope.source_device and e.key.dsl == scope.dsl and e.status == "eligible"
                          and e.fold in scope.training_folds]
    expected = {(op, dt, m) for op, dt in expected_tasks for m in models}
    present = {(r.operator, r.dtype, r.model) for r in scope.selected}
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    return {"expected": len(expected), "present": len(present & expected), "missing": missing, "extra": extra,
            "models": sorted(models), "complete": not missing and not extra}
