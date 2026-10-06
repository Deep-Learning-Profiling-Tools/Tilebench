"""Campaign assembly: task list, per-trajectory context, preflight gates,
dry-run (no API call, no GPU) and the live execution chain

    manifests -> task context -> provider (factory) -> TrajectoryRunner
              -> SubprocessEvaluator (device lock, sandbox) -> state / ledger / artifacts

shared by Base and Enhanced (Enhanced adds the frozen Optimization Skill
component and nothing else) and by both run types (formal: scored; validation:
engineering acceptance, unscored, never a distillation source)."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tilebench import hardware
from tilebench.paths import REPO_ROOT, list_operators

from tilebench.llm.v2.contracts.loader import ContractBundle, ContractError, load_contract
from tilebench.llm.v2.evaluation.adapters import AdapterNotReady, require_adapter
from tilebench.llm.v2.evaluation.fingerprint import diff as fingerprint_diff, evaluator_fingerprint
from tilebench.llm.v2.evaluation.job import EvaluationJob, build_evaluation_job
from tilebench.llm.v2.manifests import schema as ms
from tilebench.llm.v2.metrics import empirical
from tilebench.llm.v2.orchestration.identity import LLM_V2_OUTPUT_ROOT, trajectory_dir, trajectory_id
from tilebench.llm.v2.orchestration.state import SCHEMA as TRAJ_SCHEMA, TrajectoryState
from tilebench.llm.v2.prompts.renderer import TaskContext, render_initial, render_system, templates_sha256
from tilebench.llm.v2.providers.factory import GeneratorSpec, build_provider, generator_spec
from tilebench.llm.v2.providers.ledger import append_jsonl
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
    run_type: str = "formal"
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_task_context(e: Eligibility, study: dict, manifest: dict, condition: str, *,
                       require_approved: bool = True, arch: str | None = None,
                       provider: str | None = None, provider_sendable: bool = True) -> TaskContext:
    dsl_cfg = study["dsls"][e.key.dsl]
    comps = compose_context(manifest, study, dsl=e.key.dsl, device=e.key.device, fold=e.fold,
                            condition=condition, require_approved=require_approved, provider=provider,
                            provider_sendable=provider_sendable)
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
                         contract_hash: str, template_hash: str, *, run_type: str = "formal",
                         campaign: str | None = None, generator: dict | None = None,
                         evaluation_job: dict | None = None, evaluator_fp: dict | None = None,
                         scoring_binding: dict | None = None) -> TrajectoryState:
    task = {"operator": e.key.operator, "dtype": e.key.dtype, "case_id": e.key.case_id, "params": e.params,
            "device": e.key.device, "dsl": e.key.dsl, "fold": e.fold, "fp8_format": e.fp8_format,
            "problem_size": e.problem_size, "case_index": e.case_index}
    hashes = hash_record(ctx.components)
    hashes["contract"] = contract_hash
    hashes["templates"] = template_hash
    return TrajectoryState(schema=TRAJ_SCHEMA, trajectory_id=trajectory_id(task, model, condition), task=task,
                           model=model, condition=condition, config_hash=config_hash, content_hashes=hashes,
                           output_file=ctx.output_file, run_type=run_type, campaign=campaign, generator=generator,
                           evaluation_job=evaluation_job, evaluator_fingerprint=evaluator_fp,
                           scoring_binding=scoring_binding)


def preflight(device: str, dsl: str, condition: str, *, live: bool, study: dict | None = None,
              run_type: str = "formal", models_selected: tuple[str, ...] | None = None,
              provider: str | None = None, operators: list[str] | None = None) -> Preflight:
    """Gates for one (device, dsl, condition). `live=False` is the
    development report. `run_type` decides the live gates: formal requires
    approved contracts/skills/models; validation accepts drafts and
    candidate models but still requires provider grants, a ready timing
    adapter and the right host."""
    if run_type not in ms.RUN_TYPES:
        raise ValueError(f"unknown run type {run_type!r}")
    study = study or ms.load_study()
    models = ms.load_models()
    folds = ms.load_folds()
    pf = Preflight(device=device, dsl=dsl, condition=condition, ok=True, run_type=run_type)
    formal = run_type == "formal"
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
                                    require_approved=(live and formal), provider=provider if live else None,
                                    provider_sendable=live)
            pf.facts["components"] = hash_record(comps)
            pf.facts["component_status"] = {f"{c.kind}:{c.key}@{c.version}": c.status for c in comps}
            if live and not formal and any(c.status == "draft" for c in comps):
                pf.warnings.append("validation run uses draft skills (recorded; unscored)")
        except SkillError as e:
            pf.blockers.append(f"skills: {e}")
    # contracts
    n_ok, n_draft, missing = 0, 0, []
    scope = operators or list_operators()
    for op in scope:
        try:
            c = load_contract(op, require_approved=False)
            if c.status == "approved":
                n_ok += 1
            else:
                n_draft += 1
        except ContractError:
            missing.append(op)
    pf.facts["contracts"] = {"approved": n_ok, "draft_or_needs_review": n_draft, "missing": missing, "scope": len(scope)}
    if live and formal and (n_draft or missing):
        pf.blockers.append(f"contracts: {n_draft} not approved, {len(missing)} missing; formal runs need approved contracts")
    if live and not formal and missing:
        pf.blockers.append(f"contracts missing for {missing}")
    if live and not formal and n_draft:
        pf.warnings.append(f"validation run uses {n_draft} unapproved contract(s) (recorded; unscored)")
    if live:
        accept = ("approved",) if formal else ("approved", "candidate")
        pf.blockers.extend(ms.blockers_models(models, accept_status=accept, names=models_selected))
        fold_problems = ms.blockers_folds(folds)
        if formal:
            # Base trajectories are the source data of cross-fitted skill distillation: the A/B/C split must be
            # frozen before the first formal Base result exists, not only before Enhanced.
            pf.blockers.extend(f"{b} (formal Base and Enhanced both require frozen, approved folds)" for b in fold_problems)
        elif condition == "enhanced":
            pf.blockers.extend(fold_problems)
        elif fold_problems:
            pf.warnings.append("validation run with unfrozen folds (unscored; never a distillation source)")
    # scoring: declared arithmetic model (arithmetic_modes.yaml) + this device's empirical profile
    sc = scoring_gate(device, dsl, study, folds, operators)
    pf.facts["scoring"] = sc["facts"]
    if live:
        (pf.blockers if formal else pf.warnings).extend(sc["problems"])
    # device / timing adapter
    adapter = study["support_matrix"][device]["timing_adapter"]
    pf.facts["timing_adapter"] = adapter
    try:
        require_adapter(adapter)
    except AdapterNotReady as e:
        pf.blockers.append(f"{e} (NKI timing adapter is not validated on Trn2; docs/llm_v2/NKI_HANDOFF.md)")
    arch = hardware.detect_arch()
    pf.facts["detected_arch"] = arch
    expected = study["support_matrix"][device]["arch"]
    if live and arch != expected:
        pf.blockers.append(f"this host detects arch {arch!r}, campaign device {device} expects {expected!r}")
    if live:
        from tilebench.llm.v2.evaluation.launcher import detect_isolation, isolation_probe
        iso = detect_isolation("auto")
        pf.facts["isolation_backend"] = iso["backend"]
        if iso["backend"] == "bwrap":
            probe = isolation_probe()
            pf.facts["isolation_probe"] = {"ok": probe["ok"], "results": probe["results"], "device_nodes": probe["device_nodes"]}
            if formal and not probe["ok"]:
                pf.blockers.append("isolation probe failed: a restricted path is readable or a required one is not (see facts.isolation_probe)")
        elif formal:
            pf.blockers.append(f"formal runs require the bwrap sandbox; detected backend {iso['backend']!r} ({iso.get('probe_error')})")
        else:
            pf.warnings.append(f"isolation backend {iso['backend']!r}: validation run proceeds without the sandbox guarantees")
        nodes = iso["device_nodes"]
        vendor = {"blackwell": "nvidia", "hopper": "nvidia", "cdna3": "amd", "trainium2": "neuron"}.get(expected)
        if vendor and nodes.get(vendor, {}).get("verified") == "pending":
            pf.warnings.append(f"{vendor} device-node binding inside the sandbox is pending verification on this device class")
        if vendor in ("nvidia", "amd"):
            # the evaluator inherits this process's environment: a mixed BLAS stack (wheel cuBLAS + a system cuBLASLt)
            # breaks _int_mm/_scaled_mm references; formal runs fail closed on the same check the calibration uses
            from tilebench.llm.v2.calibration.environment import blas_stack_check
            blas = blas_stack_check()
            pf.facts["blas_stack"] = {k: blas.get(k) for k in ("ok", "error", "libraries", "blas_dirs", "checks", "LD_LIBRARY_PATH")}
            if not blas.get("ok"):
                (pf.blockers if formal else pf.warnings).append(
                    f"BLAS stack: {blas.get('error')} (LD_LIBRARY_PATH={blas.get('LD_LIBRARY_PATH')}); formal runs require a "
                    "consistent cuBLAS/cuBLASLt (hipBLAS/hipBLASLt) stack in the campaign environment")
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
            ctx = build_task_context(e, study, manifest, condition, require_approved=require_approved,
                                     provider_sendable=False)
            system, user = render_system(ctx), render_initial(ctx)
            row.update({"prompt_chars": len(system) + len(user), "components": hash_record(ctx.components)})
        except (SkillError, ContractError, Exception) as ex:  # noqa: BLE001
            row["error"] = f"{type(ex).__name__}: {ex}"
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# live execution chain
# --------------------------------------------------------------------------

@dataclass
class CampaignSpec:
    name: str
    run_type: str                     # formal | validation
    device: str
    dsl: str
    condition: str                    # base | enhanced
    model: str                        # generator role key: gpt | claude
    operators: list[str] | None = None
    dtypes: list[str] | None = None
    out_root: Path | None = None      # default outputs/llm_v2
    resume: bool = False
    stop_after_rounds: int | None = None
    isolation: str = "auto"           # auto | bwrap | none
    worker_timeout_s: int = 1800
    max_trajectories: int | None = None
    retry_incomplete: bool = False    # re-open a round closed by an evaluation-side infrastructure failure
    resume_transport: bool = False    # re-open an attempt closed by exhausted transport retries / a provider refusal
    resume_reason: str | None = None  # required with resume_transport (recorded as a durable `reopened` event)
    allow_evaluator_change: bool = False   # validation only: continue although the evaluator fingerprint changed (recorded)
    executor: str = "runner"          # who runs this process (recorded in META / compliance / reopen events)

    def record(self) -> dict:
        d = asdict(self)
        d["out_root"] = str(self.out_root) if self.out_root else None
        return d


def select_tasks(study: dict, folds: dict, device: str, dsl: str, operators: list[str] | None,
                 dtypes: list[str] | None) -> list[Eligibility]:
    table = [e for e in task_table(study, folds, operators) if e.key.device == device and e.key.dsl == dsl]
    if dtypes:
        table = [e for e in table if e.key.dtype in dtypes]
    return table


def _git_head() -> dict:
    out = {"sha": None, "dirty": None}
    try:
        out["sha"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True,
                                    timeout=30).stdout.strip() or None
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=str(REPO_ROOT),
                                capture_output=True, text=True, timeout=30).stdout
        out["dirty"] = bool(status.strip())
    except Exception:  # noqa: BLE001
        pass
    return out


def scoring_gate(device: str, dsl: str, study: dict, folds: dict, operators: list[str] | None) -> dict:
    """Readiness of the empirical scoring ceiling for every eligible task of
    (device, dsl): the declaration must validate (and, for formal runs, be
    approved), the device's profile must be registered, loadable and (formal)
    frozen, and every task must have status ok (no pending decision, no
    unavailable peak). Returns {"facts", "problems"}."""
    problems: list[str] = []
    facts: dict = {"binding": empirical.scoring_binding(device)}
    try:
        modes_doc = empirical.load_modes()
        errs = empirical.validate_modes(modes_doc)
        if errs:
            problems.append(f"arithmetic_modes.yaml: {errs[:3]}")
        facts["declaration_status"] = modes_doc.get("status")
        if modes_doc.get("status") != "approved":
            problems.append(f"arithmetic_modes.yaml revision {modes_doc.get('revision')} is {modes_doc.get('status')!r}; "
                            "formal scoring needs the owner's approval (decision M1)")
    except Exception as e:  # noqa: BLE001
        problems.append(f"arithmetic_modes.yaml: {e}")
        return {"facts": facts, "problems": problems}
    entry = empirical.device_entry(device)
    profile = None
    try:
        profile = empirical.load_profile(entry)
        facts["calibration_id"] = profile.get("calibration_id")
    except empirical.ProfileUnavailable as e:
        problems.append(f"empirical profile for {device}: {e}")
    if entry.get("status") != "frozen":
        problems.append(f"empirical profile for {device} is {entry.get('status')!r}; a formal campaign needs a frozen "
                        "profile (owner: `calibration-register --status frozen --by ...`)")
    rows = [e.to_dict() for e in task_table(study, folds, operators) if e.key.device == device and e.key.dsl == dsl]
    table = empirical.scoring_table(device, rows, modes_doc=modes_doc, profile=profile)
    facts["status_counts"] = table["status_counts"]
    facts["scoring_sha256"] = table["scoring_sha256"]
    pending = sorted({d for r in table["rows"] for d in r.get("pending_decisions", [])})
    facts["pending_decisions"] = pending
    not_ok = {k: v for k, v in table["status_counts"].items() if k != "ok"}
    if not_ok:
        problems.append(f"scoring targets not ready for {sum(not_ok.values())} task(s): {not_ok}"
                        + (f"; pending decisions {pending}" if pending else ""))
    return {"facts": facts, "problems": problems}


def campaign_record(spec: CampaignSpec, gen: GeneratorSpec, config_hash: str, isolation: dict,
                    template_hash: str, scoring_binding: dict | None = None, preflight_facts: dict | None = None) -> dict:
    return {"schema": "tilebench-llm-v2-campaign/1", "spec": spec.record(), "generator": gen.record(),
            "config_hash": config_hash, "templates_sha256": template_hash, "host": socket.gethostname(),
            "scoring_binding": scoring_binding, "preflight_facts": preflight_facts,
            "git": _git_head(), "isolation": isolation, "pid": os.getpid(), "started": time.time(),
            "protocol_note": ("validation runs are unscored engineering acceptance of the execution chain; "
                              "they never enter E(B) curves or distillation" if spec.run_type == "validation"
                              else "formal campaign")}


class ResumeRefused(RuntimeError):
    pass


def _check_resume(state: TrajectoryState, *, config_hash: str, content_hashes: dict, run_type: str,
                  gen: GeneratorSpec, evaluator_fp: dict | None = None, allow_evaluator_change: bool = False,
                  tdir: Path | None = None, executor: str = "runner", log=lambda s: None,
                  scoring_binding: dict | None = None) -> None:
    """A resume must run under the SAME evaluator as the trajectory so far:
    besides the study config hash, the injected prompt content and the
    generator settings, the evaluator fingerprint (job tolerance/rules/
    timing/capture policy, checker and evaluation sources, worker timeout,
    isolation backend, environment) must match. Formal: any difference
    refuses. Validation: a difference is accepted only with
    `allow_evaluator_change`, and is then recorded durably (state and
    evaluator_changes.jsonl) with the differing keys."""
    if state.config_hash != config_hash:
        raise ResumeRefused(f"{state.trajectory_id}: config hash changed ({state.config_hash[:12]} -> {config_hash[:12]}); refusing to resume")
    if state.content_hashes != content_hashes:
        diff = {k: (state.content_hashes.get(k), content_hashes.get(k)) for k in set(state.content_hashes) | set(content_hashes)
                if state.content_hashes.get(k) != content_hashes.get(k)}
        raise ResumeRefused(f"{state.trajectory_id}: injected content changed since the trajectory started: {diff}; refusing to resume")
    if state.run_type != run_type:
        raise ResumeRefused(f"{state.trajectory_id}: run type {state.run_type} != {run_type}")
    if state.generator and (state.generator.get("model_id") != gen.model_id or state.generator.get("settings") != gen.settings):
        raise ResumeRefused(f"{state.trajectory_id}: generator settings changed; refusing to resume")
    if scoring_binding is not None:
        _check_scoring_binding(state, scoring_binding, run_type)
    if evaluator_fp is None:
        return
    if state.evaluator_fingerprint is None:
        msg = f"{state.trajectory_id}: no evaluator fingerprint was recorded when this trajectory started (legacy state)"
        if run_type == "formal":
            raise ResumeRefused(msg + "; a formal resume cannot verify the evaluator")
        state.notes.append(msg + "; fingerprint recorded now on resume (validation)")
        state.evaluator_fingerprint = evaluator_fp
        return
    keys = fingerprint_diff(state.evaluator_fingerprint, evaluator_fp)
    if not keys:
        return
    msg = f"{state.trajectory_id}: evaluator changed since the trajectory started: {keys}"
    if run_type == "formal" or not allow_evaluator_change:
        raise ResumeRefused(msg + ("; formal resumes refuse evaluator changes" if run_type == "formal"
                                   else "; pass allow_evaluator_change to continue a validation run (the change is recorded)"))
    event = {"at": time.time(), "executor": executor, "keys": keys,
             "old_fingerprint_sha256": state.evaluator_fingerprint.get("fingerprint_sha256"),
             "new_fingerprint_sha256": evaluator_fp.get("fingerprint_sha256"),
             "rounds_closed": sum(1 for r in state.rounds if r.status != "pending")}
    state.evaluator_changes.append(event)
    state.evaluator_fingerprint = evaluator_fp
    state.notes.append(f"evaluator change accepted on resume (validation): {keys}")
    if tdir is not None:
        append_jsonl(tdir / "evaluator_changes.jsonl", {**event, "new_fingerprint": evaluator_fp})
    log(msg + " (validation: accepted and recorded)")


def _binding_key(b: dict | None) -> tuple:
    b = b or {}
    return (b.get("profile_sha256"), b.get("arithmetic_modes_sha256"), b.get("ceiling_basis"))


def _check_scoring_binding(state: TrajectoryState, binding: dict, run_type: str) -> None:
    """The scoring ceiling (this device's empirical profile + the declared
    arithmetic model) is pinned at creation. Formal: a changed profile or
    declaration refuses the resume (a legacy state without a binding cannot
    verify it and refuses too). Validation: recorded, never refused. A
    profile registered for ANOTHER device is not part of the binding."""
    if state.scoring_binding is None:
        msg = f"{state.trajectory_id}: no scoring binding was recorded when this trajectory started (legacy state)"
        if run_type == "formal":
            raise ResumeRefused(msg + "; a formal resume cannot verify the scoring ceiling")
        state.notes.append(msg + "; binding recorded now on resume (validation)")
        state.scoring_binding = binding
        return
    if _binding_key(state.scoring_binding) == _binding_key(binding):
        return
    diff = {k: (state.scoring_binding.get(k), binding.get(k)) for k in ("profile_sha256", "arithmetic_modes_sha256", "ceiling_basis")
            if state.scoring_binding.get(k) != binding.get(k)}
    msg = f"{state.trajectory_id}: scoring binding changed since the trajectory started: {diff}"
    if run_type == "formal":
        raise ResumeRefused(msg + "; formal resumes refuse a changed empirical profile or declaration")
    state.notes.append(msg + " (validation: recorded)")
    state.scoring_binding = binding


def retry_incomplete(state: TrajectoryState) -> str | None:
    """Re-open the last round of an `incomplete` trajectory when the failure
    was on the evaluation side (worker timeout/crash before a result) and the
    round holds a cleared candidate: the candidate is evaluated again, no
    request is made. Transport/provider failures are not re-opened here (their
    charge records must stay as they are). Returns a note or None."""
    if state.status != "incomplete" or not state.rounds:
        return None
    rec = state.rounds[-1]
    if rec.status != "infrastructure_incomplete" or not rec.attempts or rec.attempts[-1].verdict != "clear":
        return None
    note = (f"round {rec.round}: infrastructure_incomplete ({(rec.diagnostic or '')[:120]!r}) re-opened for "
            f"re-evaluation on {time.strftime('%Y-%m-%d %H:%M:%S')}; no new request")
    rec.evaluation_reason = f"retry_incomplete: {(rec.diagnostic or '')[:160]}"
    rec.status, rec.diagnostic, rec.evaluation = "pending", None, None
    rec.latency_ms_mean = rec.latency_ms_samples = None
    state.status, state.stop_reason = "in_progress", None
    state.notes.append(note)
    return note


def reopen_transport_attempt(state: TrajectoryState, tdir: Path, *, reason: str, executor: str,
                             settings_hash_before: str | None = None, settings_hash_after: str | None = None) -> str | None:
    """Explicit, recorded request-side resume: re-open the last round's
    attempt that was closed by exhausted transport retries or by a provider
    refusal. The attempt record is moved to `reopened_attempts` (nothing is
    deleted), a durable `reopened` event is appended to transport.jsonl, and
    the next generation of the SAME round/attempt rebuilds its transport
    history from every durable event, so earlier unknown charges keep
    propagating. A provider refusal (configuration / authentication error)
    is re-opened only with a reason; the generator settings hash before and
    after are recorded so a corrected configuration is a visible new
    version, never a silent repeat. No optimization round or candidate is
    added by this operation."""
    if state.status != "incomplete" or not state.rounds:
        return None
    rec = state.rounds[-1]
    if rec.status != "infrastructure_incomplete" or not rec.attempts:
        return None
    last = rec.attempts[-1]
    if last.verdict not in ("transport_failed", "provider_refused"):
        return None
    if not reason or not reason.strip():
        raise ResumeRefused("resume_transport needs a reason")
    if last.verdict == "provider_refused" and settings_hash_before == settings_hash_after:
        state.notes.append(f"round {rec.round} attempt {last.attempt}: provider refusal re-opened with unchanged settings "
                           f"(reason: {reason.strip()}); the provider's rejection may repeat")
    removed = rec.attempts.pop()
    rec.reopened_attempts.append({"attempt": removed.attempt, "verdict": removed.verdict, "cost": removed.cost,
                                  "cost_status": removed.cost_status, "transport": removed.transport, "error": removed.error,
                                  "reopened_at": time.time(), "reason": reason, "executor": executor})
    event = {"round": rec.round, "attempt": removed.attempt, "event": "reopened", "previous_verdict": removed.verdict,
             "previous_cost_status": removed.cost_status, "reason": reason, "executor": executor,
             "settings_hash_before": settings_hash_before, "settings_hash_after": settings_hash_after, "t": time.time()}
    append_jsonl(tdir / "transport.jsonl", event)
    rec.status, rec.diagnostic = "pending", None
    state.status, state.stop_reason = "in_progress", None
    note = (f"round {rec.round} attempt {removed.attempt}: {removed.verdict} re-opened by {executor} ({reason.strip()}); "
            "prior transport charges are rebuilt from transport.jsonl")
    state.notes.append(note)
    return note


def run_campaign(spec: CampaignSpec, *, log=print, sleep=time.sleep) -> dict:
    """The live chain. Refuses on any preflight blocker. Returns a summary."""
    from tilebench.llm.v2.evaluation.launcher import SubprocessEvaluator
    from tilebench.llm.v2.orchestration.runner import RunnerConfig, TrajectoryRunner

    study, models, folds, modes = ms.load_study(), ms.load_models(), ms.load_folds(), ms.load_arithmetic_modes()
    formal = spec.run_type == "formal"
    accept = ("approved",) if formal else ("approved", "candidate")
    gen = generator_spec(models, spec.model, accept_status=accept)
    pf = preflight(spec.device, spec.dsl, spec.condition, live=True, study=study, run_type=spec.run_type,
                   models_selected=(spec.model,), provider=gen.provider, operators=spec.operators)
    if not pf.ok:
        return {"refused": True, "preflight": pf.to_dict()}
    frozen_timeout = int((study.get("evaluation") or {}).get("worker_timeout_s", spec.worker_timeout_s))
    if formal and spec.worker_timeout_s != frozen_timeout:
        return {"refused": True, "preflight": pf.to_dict(),
                "reason": f"formal runs use the frozen per-candidate wall-clock limit {frozen_timeout} s "
                          f"(study.yaml evaluation.worker_timeout_s), identical for every model and DSL; got {spec.worker_timeout_s}"}
    config_hash = ms.study_config_hash(study, models, folds, modes)
    template_hash = templates_sha256()
    manifest = load_manifest()
    arch = hardware.detect_arch()
    out_root = spec.out_root or LLM_V2_OUTPUT_ROOT
    cdir = out_root / spec.name
    cdir.mkdir(parents=True, exist_ok=True)
    provider = build_provider(gen, timeout_s=float(models.get("transport", {}).get("timeout_s", 3600)))
    evaluator = SubprocessEvaluator(device=spec.device, timeout_s=spec.worker_timeout_s,
                                    sandbox_root=cdir / "_sandbox", isolation=spec.isolation, lock_root=out_root)
    if formal and evaluator.report.get("backend") != "bwrap":
        return {"refused": True, "preflight": pf.to_dict(),
                "reason": f"formal runs require the bwrap sandbox; isolation backend is {evaluator.report.get('backend')!r}"}
    binding = empirical.scoring_binding(spec.device)
    rec = campaign_record(spec, gen, config_hash, evaluator.report, template_hash, binding,
                          preflight_facts={k: pf.facts.get(k) for k in ("blas_stack", "scoring", "contracts", "component_status",
                                                                        "components", "isolation_probe", "detected_arch")})
    first = cdir / f"campaign_{spec.condition}_{spec.dsl}_{spec.model}.json"
    if not first.exists():
        first.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    with open(cdir / "campaign_runs.jsonl", "a") as fh:      # one record per process start (resumes included)
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    tasks = [e for e in select_tasks(study, folds, spec.device, spec.dsl, spec.operators, spec.dtypes)]
    if spec.max_trajectories:
        tasks = tasks[:spec.max_trajectories]
    transport = models.get("transport", {})
    cfg = RunnerConfig(model_id=gen.model_id, provider_name=gen.provider, settings=gen.settings,
                       rounds=study["trajectory"]["rounds"], max_generations=study["trajectory"]["max_generations_per_round"],
                       max_transport_retries=int(transport.get("max_transport_retries", 3)),
                       retry_backoff_s=float(transport.get("retry_backoff_s", 30)), feedback_limits=study["feedback"],
                       total_prompt_max_chars=study["context_limits"]["total_prompt"], executor=spec.executor)
    summary = {"campaign": spec.name, "run_type": spec.run_type, "generator": gen.record(), "config_hash": config_hash,
               "scoring_binding": binding, "isolation": evaluator.report, "trajectories": [], "skipped": []}
    for e in tasks:
        if e.status != "eligible":
            summary["skipped"].append({"task": e.key.as_str(), "status": e.status, "reason": e.reason})
            continue
        contract: ContractBundle = load_contract(e.key.operator, require_approved=formal)
        ctx = build_task_context(e, study, manifest, spec.condition, require_approved=formal, arch=arch,
                                 provider=gen.provider)
        identity = {"campaign": spec.name, "run_type": spec.run_type, "model": spec.model, "condition": spec.condition}
        job: EvaluationJob = build_evaluation_job(operator=e.key.operator, dtype=e.key.dtype, params=e.params,
                                                  dsl=e.key.dsl, device=e.key.device, arch=arch, rules=contract.rules,
                                                  study=study, identity=identity)
        if (ctx.atol, ctx.rtol) != (job.atol, job.rtol):
            raise RuntimeError(f"{e.key.as_str()}: prompt tolerance {ctx.atol}/{ctx.rtol} != evaluator tolerance {job.atol}/{job.rtol}")
        task = {"operator": e.key.operator, "dtype": e.key.dtype, "case_id": e.key.case_id, "params": e.params,
                "device": e.key.device, "dsl": e.key.dsl}
        tdir = trajectory_dir(spec.name, task, spec.model, spec.condition, root=out_root)
        tpath = tdir / "trajectory.json"
        content_hashes = {**hash_record(ctx.components), "contract": contract.sha256, "templates": template_hash}
        fp = evaluator_fingerprint(job, worker_timeout_s=spec.worker_timeout_s, isolation_backend=evaluator.report.get("backend", "none"))
        if tpath.exists():
            if not spec.resume:
                raise RuntimeError(f"{tdir} already holds a trajectory; pass resume=True to continue it")
            state = TrajectoryState.load(tpath)
            _check_resume(state, config_hash=config_hash, content_hashes=content_hashes, run_type=spec.run_type, gen=gen,
                          evaluator_fp=fp, allow_evaluator_change=spec.allow_evaluator_change, tdir=tdir,
                          executor=spec.executor, log=lambda s, k=e.key.as_str(): log(f"[{k}] {s}"),
                          scoring_binding=binding)
            job.identity["trajectory_id"] = state.trajectory_id
            if spec.retry_incomplete:
                note = retry_incomplete(state)
                if note:
                    log(f"[{e.key.as_str()}] {note}")
            if spec.resume_transport:
                shash = ms.sha256_text(ms.canonical_json(gen.settings))
                prev = ms.sha256_text(ms.canonical_json((state.generator or {}).get("settings"))) if state.generator else None
                note = reopen_transport_attempt(state, tdir, reason=spec.resume_reason or "", executor=spec.executor,
                                                settings_hash_before=prev, settings_hash_after=shash)
                if note:
                    log(f"[{e.key.as_str()}] {note}")
            log(f"[{e.key.as_str()}] resuming {state.trajectory_id}: status {state.status}, "
                f"{sum(1 for r in state.rounds if r.status != 'pending')} rounds closed, {state.request_count()} transport events")
        else:
            state = new_trajectory_state(e, ctx, spec.model, spec.condition, config_hash, contract.sha256, template_hash,
                                         run_type=spec.run_type, campaign=spec.name, generator=gen.record(),
                                         evaluation_job=job.record(), evaluator_fp=fp, scoring_binding=binding)
            job.identity["trajectory_id"] = state.trajectory_id
            state.evaluation_job = job.record()
            log(f"[{e.key.as_str()}] new trajectory {state.trajectory_id} -> {tdir} (evaluator fingerprint {fp['fingerprint_sha256'][:12]})")
        runner = TrajectoryRunner(state=state, tdir=tdir, ctx=ctx, provider=provider, evaluator=evaluator, job=job,
                                  cfg=cfg, sleep=sleep, log=lambda s, k=e.key.as_str(): log(f"[{k}] {s}"))
        status = runner.run(stop_after_rounds=spec.stop_after_rounds)
        summary["trajectories"].append(trajectory_summary(state, tdir, status))
        log(f"[{e.key.as_str()}] -> {status}")
    (cdir / f"summary_{spec.model}_{spec.dsl}_{int(time.time())}.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return summary


def trajectory_summary(state: TrajectoryState, tdir: Path, status: str | None = None) -> dict:
    rounds = [{"round": r.round, "status": r.status, "attempts": len(r.attempts),
               "verdicts": [a.verdict for a in r.attempts], "latency_ms_mean": r.latency_ms_mean,
               "latency_ms_samples": r.latency_ms_samples, "cost": [a.cost for a in r.attempts],
               "response_ids": [a.response_id for a in r.attempts], "timing_execution_mode": r.timing_execution_mode}
              for r in state.rounds]
    costs = [a.cost for r in state.rounds for a in r.attempts]
    return {"trajectory_id": state.trajectory_id, "dir": str(tdir), "task": state.task, "model": state.model,
            "condition": state.condition, "run_type": state.run_type, "status": status or state.status,
            "state_status": state.status, "stop_reason": state.stop_reason, "rounds_closed": sum(1 for r in state.rounds if r.status != "pending"),
            "valid_rounds": len(state.valid_rounds()), "attempts": sum(len(r.attempts) for r in state.rounds),
            "transport_events": state.request_count(),
            "cumulative_tokens": None if any(c is None for c in costs) else sum(costs),
            "cost_exact": not any(c is None for c in costs), "best_valid": state.best_valid, "rounds": rounds,
            "notes": state.notes}
