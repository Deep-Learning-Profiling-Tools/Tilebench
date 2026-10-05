"""Loading, validation and hashing of the frozen study manifests.

Every manifest is plain YAML next to this file. `validate_*` functions raise
ManifestError with a precise message; `blockers_*` functions return the list
of reasons a LIVE campaign must not start (they never block mock tests)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

MANIFEST_DIR = Path(__file__).resolve().parent

CONDITIONS = ("base", "enhanced")
FOLDS = ("A", "B", "C")
DSLS = ("triton", "cutile", "tilelang", "nki")
DEVICES = ("B200", "GH200", "MI300X", "Trn2")


class ManifestError(ValueError):
    pass


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(*parts: Any) -> str:
    """Stable hash of one or more JSON-able objects (manifests, task specs)."""
    return sha256_text(canonical_json(list(parts)))


def load_yaml(name: str) -> dict:
    path = MANIFEST_DIR / name
    if not path.exists():
        raise ManifestError(f"manifest {name} is missing at {path}")
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ManifestError(f"manifest {name} must be a mapping")
    return data


def load_study() -> dict:
    study = load_yaml("study.yaml")
    validate_study(study)
    return study


def load_models() -> dict:
    return load_yaml("models.yaml")


def load_folds() -> dict:
    return load_yaml("folds.yaml")


def load_arithmetic_modes() -> dict:
    return load_yaml("arithmetic_modes.yaml")


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def validate_study(study: dict) -> None:
    if study.get("protocol") != "tilebench-llm-v2" or study.get("revision") != 2:
        raise ManifestError("study.yaml: protocol/revision must be tilebench-llm-v2 / 2")
    traj = study.get("trajectory", {})
    if traj.get("rounds") != 10:
        raise ManifestError("study.yaml: trajectory.rounds must be 10")
    if traj.get("max_generations_per_round") != 3:
        raise ManifestError("study.yaml: trajectory.max_generations_per_round must be 3")
    timing = study.get("timing", {})
    if (timing.get("warmup"), timing.get("repeat")) != (1, 3):
        raise ManifestError("study.yaml: timing must be warmup=1, repeat=3")
    conds = study.get("conditions", {})
    if set(conds) != set(CONDITIONS):
        raise ManifestError(f"study.yaml: conditions must be exactly {CONDITIONS}")
    if conds["enhanced"] != conds["base"] + ["optimization_skill"]:
        raise ManifestError("study.yaml: enhanced must equal base + [optimization_skill]")
    matrix = study.get("support_matrix", {})
    if set(matrix) != set(DEVICES):
        raise ManifestError(f"study.yaml: support_matrix must list exactly {DEVICES}")
    dsls = study.get("dsls", {})
    if set(dsls) != set(DSLS):
        raise ManifestError(f"study.yaml: dsls must list exactly {DSLS}")
    for dev, entry in matrix.items():
        unknown = set(entry.get("dsls", [])) - set(DSLS)
        if unknown:
            raise ManifestError(f"study.yaml: support_matrix[{dev}] names unknown DSLs {sorted(unknown)}")
    transfer = study.get("skill_transfer", {})
    for dsl, devices in transfer.items():
        for dev in devices:
            if dsl not in matrix.get(dev, {}).get("dsls", []):
                raise ManifestError(f"study.yaml: skill_transfer[{dsl}] includes {dev}, which does not support it")
        src = dsls[dsl]["source_device"]
        if src not in devices:
            raise ManifestError(f"study.yaml: skill_transfer[{dsl}] must include its source device {src}")
    if list(study.get("folds", [])) != list(FOLDS):
        raise ManifestError("study.yaml: folds must be [A, B, C]")
    for key in ("reference_skill", "device_context_skill", "algorithm_contract",
                "optimization_skill", "functional_reference", "total_prompt"):
        if not isinstance(study.get("context_limits", {}).get(key), int):
            raise ManifestError(f"study.yaml: context_limits.{key} must be an integer")


def validate_folds(folds: dict, operators: list[str]) -> None:
    if folds.get("status") not in ("proposed", "frozen"):
        raise ManifestError("folds.yaml: status must be proposed or frozen")
    if folds.get("status") == "frozen" and not folds.get("approved_by"):
        raise ManifestError("folds.yaml: a frozen fold assignment needs approved_by")
    assigned: dict[str, str] = {}
    for fold in FOLDS:
        ops = folds.get("folds", {}).get(fold, {}).get("operators")
        if not ops:
            raise ManifestError(f"folds.yaml: fold {fold} has no operators")
        for op in ops:
            if op in assigned:
                raise ManifestError(f"folds.yaml: operator {op} is in folds {assigned[op]} and {fold}")
            assigned[op] = fold
    missing = sorted(set(operators) - set(assigned))
    extra = sorted(set(assigned) - set(operators))
    if missing or extra:
        raise ManifestError(f"folds.yaml: missing operators {missing}; unknown operators {extra}")
    fam_members = [op for members in folds.get("families", {}).values() for op in members]
    if sorted(fam_members) != sorted(operators):
        raise ManifestError("folds.yaml: families must partition the operator set exactly")
    for fam, members in folds["families"].items():
        fold_set = {assigned[m] for m in members}
        if len(fold_set) != 1:
            raise ManifestError(f"folds.yaml: family {fam} is split across folds {sorted(fold_set)}")


def fold_of(folds: dict, operator: str) -> str:
    for fold in FOLDS:
        if operator in folds["folds"][fold]["operators"]:
            return fold
    raise ManifestError(f"operator {operator} has no fold")


def training_folds(held_out: str) -> tuple[str, ...]:
    if held_out not in FOLDS:
        raise ManifestError(f"unknown fold {held_out}")
    return tuple(f for f in FOLDS if f != held_out)


def blockers_models(models: dict, roles: tuple[str, ...] = ("generator",)) -> list[str]:
    """Reasons a live campaign cannot start: unset model ids / settings."""
    out: list[str] = []
    for role in roles:
        spec = models.get("roles", {}).get(role)
        if spec is None:
            out.append(f"models.yaml: role {role} missing")
            continue
        entries = spec.items() if role == "generator" else [(role, spec)]
        for name, cfg in entries:
            if role == "reviewer" and not cfg.get("enabled"):
                continue
            if not cfg.get("model_id"):
                out.append(f"models.yaml: {role}.{name}.model_id is not set")
            if not cfg.get("provider"):
                out.append(f"models.yaml: {role}.{name}.provider is not set")
            if cfg.get("status") != "approved":
                out.append(f"models.yaml: {role}.{name}.status is {cfg.get('status')!r}, not approved")
    return out


def blockers_folds(folds: dict) -> list[str]:
    return [] if folds.get("status") == "frozen" else ["folds.yaml: fold assignment is not frozen"]


def study_config_hash(study: dict, models: dict, folds: dict, modes: dict) -> str:
    """Hash of everything that defines a campaign's protocol. Trajectories
    carry it; the runner refuses to resume under a different hash."""
    return config_hash(study, models, folds, modes)
