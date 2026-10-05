"""Campaign assembly: task list, per-trajectory context, preflight gates and
dry-run (no API call, no GPU)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tilebench import hardware
from tilebench.paths import list_operators

from tilebench.llm.v2.contracts.loader import ContractError, load_contract
from tilebench.llm.v2.manifests import schema as ms
from tilebench.llm.v2.orchestration.identity import trajectory_dir, trajectory_id
from tilebench.llm.v2.orchestration.state import SCHEMA as TRAJ_SCHEMA, TrajectoryState
from tilebench.llm.v2.prompts.renderer import TaskContext, render_initial, render_system
from tilebench.llm.v2.skills.loader import SkillError, compose_context, hash_record, load_manifest
from tilebench.llm.v2.tasks.case_selection import load_operator_config
from tilebench.llm.v2.tasks.fields import task_fields
from tilebench.llm.v2.tasks.support import Eligibility, task_table


@dataclass
class Preflight:
    device: str
    dsl: str
    condition: str
    ok: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_task_context(e: Eligibility, study: dict, manifest: dict, condition: str, *,
                       require_approved: bool = True, arch: str | None = None) -> TaskContext:
    dsl_cfg = study["dsls"][e.key.dsl]
    comps = compose_context(manifest, study, dsl=e.key.dsl, device=e.key.device, fold=e.fold,
                            condition=condition, require_approved=require_approved)
    contract = load_contract(e.key.operator, require_approved=require_approved)
    cfg = load_operator_config(e.key.operator)
    tf = task_fields(e.key.operator, e.key.dtype, e.params, cfg, arch)
    if len(tf.functional_reference.source) > study["context_limits"]["functional_reference"]:
        raise SkillError(f"{e.key.operator}: functional reference exceeds the context limit")
    if len(contract.model_text) > study["context_limits"]["algorithm_contract"]:
        raise SkillError(f"{e.key.operator}: contract exceeds the context limit")
    return TaskContext(
        operator=e.key.operator, dtype=e.key.dtype, torch_dtype=tf.torch_dtype, dsl=e.key.dsl,
        dsl_version=dsl_cfg["reference_version"], device=e.key.device, output_file=dsl_cfg["output_file"],
        params=e.params, atol=tf.tolerance["atol"], rtol=tf.tolerance["rtol"], tolerance_source=tf.tolerance["source"],
        run_signature=tf.run_signature, returns=tf.returns, functional_reference=tf.functional_reference.source,
        fp8_format=e.fp8_format, components=comps, contract_text=contract.model_text,
        rounds=study["trajectory"]["rounds"],
    )


def new_trajectory_state(e: Eligibility, ctx: TaskContext, model: str, condition: str, config_hash: str,
                         contract_hash: str, template_hash: str) -> TrajectoryState:
    task = {"operator": e.key.operator, "dtype": e.key.dtype, "case_id": e.key.case_id, "params": e.params,
            "device": e.key.device, "dsl": e.key.dsl, "fold": e.fold, "fp8_format": e.fp8_format,
            "problem_size": e.problem_size, "case_index": e.case_index}
    hashes = hash_record(ctx.components)
    hashes["contract"] = contract_hash
    hashes["templates"] = template_hash
    return TrajectoryState(schema=TRAJ_SCHEMA, trajectory_id=trajectory_id(task, model, condition), task=task,
                           model=model, condition=condition, config_hash=config_hash, content_hashes=hashes,
                           output_file=ctx.output_file)


def preflight(device: str, dsl: str, condition: str, *, live: bool, study: dict | None = None) -> Preflight:
    study = study or ms.load_study()
    models = ms.load_models()
    folds = ms.load_folds()
    modes = ms.load_arithmetic_modes()
    pf = Preflight(device=device, dsl=dsl, condition=condition, ok=True)
    try:
        ms.validate_folds(folds, list_operators())
    except ms.ManifestError as e:
        pf.blockers.append(str(e))
    if dsl not in study["support_matrix"][device]["dsls"]:
        pf.blockers.append(f"{dsl} is not supported on {device} (support matrix)")
    try:
        manifest = load_manifest()
    except SkillError as e:
        pf.blockers.append(str(e))
        manifest = None
    if manifest is not None:
        try:
            comps = compose_context(manifest, study, dsl=dsl, device=device, fold="A", condition=condition,
                                    require_approved=live)
            pf.facts["components"] = hash_record(comps)
        except SkillError as e:
            pf.blockers.append(f"skills: {e}")
    # contracts
    n_ok, n_draft, missing = 0, 0, []
    for op in list_operators():
        try:
            c = load_contract(op, require_approved=False)
            if c.status == "approved":
                n_ok += 1
            else:
                n_draft += 1
        except ContractError:
            missing.append(op)
    pf.facts["contracts"] = {"approved": n_ok, "draft": n_draft, "missing": missing}
    if live and (n_draft or missing):
        pf.blockers.append(f"contracts: {n_draft} draft, {len(missing)} missing; live runs need approved contracts")
    if live:
        pf.blockers.extend(ms.blockers_models(models))
        if condition == "enhanced":
            pf.blockers.extend(ms.blockers_folds(folds))
    # device / timing adapter
    adapter = study["support_matrix"][device]["timing_adapter"]
    pf.facts["timing_adapter"] = adapter
    if adapter == "neuron_runtime_trace":
        pf.blockers.append("NKI timing adapter is not validated on Trn2 (docs/llm_v2/NKI_HANDOFF.md)")
    arch = hardware.detect_arch()
    pf.facts["detected_arch"] = arch
    expected = study["support_matrix"][device]["arch"]
    if live and arch != expected:
        pf.blockers.append(f"this host detects arch {arch!r}, campaign device {device} expects {expected!r}")
    pf.ok = not pf.blockers
    return pf


def dry_run(device: str, dsl: str, condition: str, *, operators: list[str] | None = None,
            require_approved: bool = False, limit: int | None = None) -> list[dict]:
    """Render the initial prompt of every eligible task without any API call."""
    study, folds = ms.load_study(), ms.load_folds()
    manifest = load_manifest()
    rows = []
    table = [e for e in task_table(study, folds, operators) if e.key.device == device and e.key.dsl == dsl]
    for e in table[:limit] if limit else table:
        row = {"task": e.key.as_str(), "status": e.status, "fold": e.fold}
        if e.status != "eligible":
            row["skipped"] = e.reason
            rows.append(row)
            continue
        try:
            ctx = build_task_context(e, study, manifest, condition, require_approved=require_approved)
            system, user = render_system(ctx), render_initial(ctx)
            row.update({"prompt_chars": len(system) + len(user), "components": hash_record(ctx.components)})
        except (SkillError, ContractError, Exception) as ex:  # noqa: BLE001
            row["error"] = f"{type(ex).__name__}: {ex}"
        rows.append(row)
    return rows
