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
# What the timing adapter does when a requested CUDA-graph capture fails
# (study.yaml:timing.capture_failure_policy). Only one policy is defined:
# the candidate is timed eagerly, the record says so, and every consumer
# (metrics, publication) flags the round as measured in a different mode.
CAPTURE_FAILURE_POLICIES = ("time_eagerly_and_flag", "timing_error")
# formal: scored Base/Enhanced campaigns (approved assets, approved models,
# frozen folds for Enhanced). validation: engineering acceptance of the
# execution chain; identical mechanics, unscored, never a distillation source.
RUN_TYPES = ("formal", "validation")
# Protocol revision 4 (study-owner decision 2026-10-07): one representative
# dtype per operator, all 20 configured cases per candidate, 5 rounds, exactly
# one candidate generation per round, Claude Code compliance adjudication.
PROTOCOL_REVISION = 4
ROUNDS = 5
MAX_GENERATIONS_PER_ROUND = 1
CASES_PER_TASK = 20
COMPLIANCE_VERDICTS = ("clear", "audit_only", "review_required", "confirmed_violation")
ADJUDICATOR = "claude-code"


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
    if study.get("protocol") != "tilebench-llm-v2" or study.get("revision") != PROTOCOL_REVISION:
        raise ManifestError(f"study.yaml: protocol/revision must be tilebench-llm-v2 / {PROTOCOL_REVISION}")
    traj = study.get("trajectory", {})
    if traj.get("rounds") != ROUNDS:
        raise ManifestError(f"study.yaml: trajectory.rounds must be {ROUNDS}")
    if traj.get("max_generations_per_round") != MAX_GENERATIONS_PER_ROUND:
        raise ManifestError(f"study.yaml: trajectory.max_generations_per_round must be {MAX_GENERATIONS_PER_ROUND} "
                            "(one round == one candidate generation; no same-round repair)")
    if "repair_only_on" in traj:
        raise ManifestError("study.yaml: revision 3 has no same-round repair (repair_only_on must not be set)")
    unit = study.get("task_unit") or {}
    if unit.get("kind") != "representative_dtype" or unit.get("manifest") != "representative_dtypes.yaml":
        raise ManifestError("study.yaml: task_unit must be {kind: representative_dtype, manifest: representative_dtypes.yaml}")
    if (unit.get("cases"), unit.get("case_manifest"), unit.get("cases_per_task"), unit.get("round_validity")) != \
            ("all_configured", "case_sets.yaml", CASES_PER_TASK, "all_cases_valid"):
        raise ManifestError(f"study.yaml: task_unit must evaluate all_configured cases (case_manifest case_sets.yaml, "
                            f"cases_per_task {CASES_PER_TASK}, round_validity all_cases_valid)")
    comp = study.get("compliance") or {}
    if comp.get("verdicts") != list(COMPLIANCE_VERDICTS) or comp.get("blocking") != ["review_required"] \
            or comp.get("missing_required_evidence") != "audit_only":
        raise ManifestError(f"study.yaml: compliance must declare verdicts {list(COMPLIANCE_VERDICTS)}, blocking "
                            "[review_required] and missing_required_evidence: audit_only (checker v2)")
    if comp.get("adjudicator") != ADJUDICATOR:
        raise ManifestError(f"study.yaml: compliance.adjudicator must be {ADJUDICATOR!r} (revision 4)")
    timing = study.get("timing", {})
    if (timing.get("warmup"), timing.get("repeat")) != (1, 3):
        raise ManifestError("study.yaml: timing must be warmup=1, repeat=3")
    if timing.get("cache_flush") != "operator_boundary_outside_timed_scope":
        raise ManifestError("study.yaml: timing.cache_flush must be operator_boundary_outside_timed_scope")
    if not isinstance(timing.get("cuda_graph_requested"), bool):
        raise ManifestError("study.yaml: timing.cuda_graph_requested must be a boolean")
    if timing.get("capture_failure_policy") not in CAPTURE_FAILURE_POLICIES:
        raise ManifestError(f"study.yaml: timing.capture_failure_policy must be one of {CAPTURE_FAILURE_POLICIES}")
    if timing.get("capture_failure_policy_formal") is not None and timing["capture_failure_policy_formal"] not in CAPTURE_FAILURE_POLICIES:
        raise ManifestError(f"study.yaml: timing.capture_failure_policy_formal must be one of {CAPTURE_FAILURE_POLICIES}")
    ev = study.get("evaluation") or {}
    if ev and (not isinstance(ev.get("worker_timeout_s"), int) or isinstance(ev.get("worker_timeout_s"), bool) or ev["worker_timeout_s"] <= 0):
        raise ManifestError("study.yaml: evaluation.worker_timeout_s must be a positive integer (frozen per-candidate wall-clock limit)")
    tb = study.get("torch_baselines") or {}
    if set(tb) - set(DEVICES):
        raise ManifestError(f"study.yaml: torch_baselines names unknown devices {sorted(set(tb) - set(DEVICES))}")
    snaps = study.get("device_snapshots") or {}
    if set(snaps) - set(DEVICES):
        raise ManifestError(f"study.yaml: device_snapshots names unknown devices {sorted(set(snaps) - set(DEVICES))}")
    if study.get("run_types") is None or set(study["run_types"]) != set(RUN_TYPES):
        raise ManifestError(f"study.yaml: run_types must define exactly {RUN_TYPES}")
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


def blockers_models(models: dict, roles: tuple[str, ...] = ("generator",), *,
                    accept_status: tuple[str, ...] = ("approved",), names: tuple[str, ...] | None = None) -> list[str]:
    """Reasons a live campaign cannot start: unset model ids / settings, or a
    status outside `accept_status` (formal runs accept only `approved`;
    validation runs also accept `candidate`). `names` restricts the generator
    entries checked to the models actually selected for the run."""
    out: list[str] = []
    for role in roles:
        spec = models.get("roles", {}).get(role)
        if spec is None:
            out.append(f"models.yaml: role {role} missing")
            continue
        entries = spec.items() if role == "generator" else [(role, spec)]
        for name, cfg in entries:
            if role == "generator" and names is not None and name not in names:
                continue
            if role == "reviewer" and not cfg.get("enabled"):
                continue
            if not cfg.get("model_id"):
                out.append(f"models.yaml: {role}.{name}.model_id is not set")
            if not cfg.get("provider"):
                out.append(f"models.yaml: {role}.{name}.provider is not set")
            if cfg.get("status") not in accept_status:
                out.append(f"models.yaml: {role}.{name}.status is {cfg.get('status')!r}, accepted {accept_status}")
    return out


def blockers_folds(folds: dict) -> list[str]:
    if folds.get("status") != "frozen":
        return ["folds.yaml: fold assignment is not frozen"]
    if not folds.get("approved_by"):
        return ["folds.yaml: fold assignment is frozen without approved_by"]
    return []


def load_representative() -> dict:
    from tilebench.llm.v2.tasks.representative import load_manifest
    return load_manifest()


def load_case_sets() -> dict:
    from tilebench.llm.v2.tasks.case_sets import load_manifest
    return load_manifest()


def load_excluded_campaigns() -> dict:
    data = load_yaml("excluded_campaigns.yaml")
    if data.get("schema") != "tilebench-llm-excluded-campaigns/1":
        raise ManifestError("excluded_campaigns.yaml: schema must be tilebench-llm-excluded-campaigns/1")
    for name, entry in (data.get("campaigns") or {}).items():
        if not entry.get("forbidden_uses") or not entry.get("decided_by"):
            raise ManifestError(f"excluded_campaigns.yaml: {name} needs forbidden_uses and decided_by")
    return data


def excluded_campaign(name: str | None) -> dict | None:
    """The exclusion record of a campaign, or None when it may be used."""
    if not name:
        return None
    return (load_excluded_campaigns().get("campaigns") or {}).get(name)


def study_config_hash(study: dict, models: dict, folds: dict, modes: dict, representative: dict | None = None,
                      case_sets: dict | None = None) -> str:
    """Hash of everything that defines a campaign's protocol (study incl. the
    5-round / one-generation rules and the checker-v2 policy, models, folds,
    arithmetic modes, the representative-dtype manifest and the frozen 20-case
    sets). Trajectories carry it; the runner refuses to resume under a
    different hash."""
    representative = representative if representative is not None else load_representative()
    case_sets = case_sets if case_sets is not None else load_case_sets()
    return config_hash(study, models, folds, modes, representative, case_sets)
