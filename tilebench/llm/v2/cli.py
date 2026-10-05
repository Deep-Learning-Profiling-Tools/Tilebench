"""Turnkey CLI: python -m tilebench.llm.v2 <command> [...]

Live commands (`base`, `enhanced`, `distill`, `probe-provider`) run the
preflight first and refuse to start on any blocker. No command installs
packages, edits settings or falls back to another device's configuration.
Keys are read from the environment by the provider adapters only."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tilebench.paths import list_operators

from tilebench.llm.v2 import PROTOCOL
from tilebench.llm.v2.manifests import schema as ms


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, sort_keys=True, default=str))


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
def cmd_doctor(a) -> int:
    import importlib.metadata as md
    from tilebench import hardware, provenance
    out = {"protocol": PROTOCOL, "python": sys.version.split()[0], "packages": {}, "device": None}
    for pkg in ("torch", "triton", "cuda-tile", "tilelang", "apache-tvm-ffi", "openai", "anthropic", "neuronx-cc"):
        try:
            out["packages"][pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out["packages"][pkg] = None
    try:
        out["device"] = {"arch": hardware.detect_arch(), "info": hardware.device_info()._asdict() if hardware.device_info() else None,
                         "llc_bytes": hardware.last_level_cache_bytes()}
    except Exception as e:  # noqa: BLE001
        out["device"] = {"error": str(e)}
    out["source"] = provenance.source_state()
    out["manifests"] = {}
    try:
        study = ms.load_study(); folds = ms.load_folds()
        ms.validate_folds(folds, list_operators())
        out["manifests"]["study"] = "ok"; out["manifests"]["folds"] = folds["status"]
        out["manifests"]["config_hash"] = ms.study_config_hash(study, ms.load_models(), folds, ms.load_arithmetic_modes())
        out["manifests"]["model_blockers_formal"] = ms.blockers_models(ms.load_models())
        out["manifests"]["model_blockers_validation"] = ms.blockers_models(ms.load_models(), accept_status=("approved", "candidate"))
    except ms.ManifestError as e:
        out["manifests"]["error"] = str(e)
    from tilebench.llm.v2.skills.loader import SkillError, load_manifest
    try:
        m = load_manifest()
        out["skills"] = {k: {key: sorted(v) for key, v in m.get(k, {}).items()} for k in ("reference", "device", "optimization")}
    except SkillError as e:
        out["skills"] = {"error": str(e)}
    from tilebench.llm.v2.contracts.loader import audit_summary
    out["contracts"] = {k: (len(v) if isinstance(v, list) else v) for k, v in audit_summary().items()}
    from tilebench.llm.v2.evaluation.launcher import detect_isolation
    out["isolation"] = detect_isolation("auto")
    _print(out)
    return 0


def cmd_inventory(a) -> int:
    from tilebench.llm.v2.skills.loader import load_manifest
    m = load_manifest()
    rows = []
    for kind in ("reference", "device", "optimization", "contract"):
        for key, versions in m.get(kind, {}).items():
            for ver, e in versions.items():
                rows.append({"kind": kind, "key": key, "version": ver, "path": e.get("path"), "status": e.get("status"),
                             "permission": e.get("permission"), "sendable_to": e.get("sendable_to", []),
                             "publishable": e.get("publishable", False),
                             "sha256_injected": (e.get("sha256_injected") or "")[:16], "chars": e.get("chars")})
    _print(rows)
    return 0


def cmd_validate_manifests(a) -> int:
    errors = []
    try:
        study = ms.load_study()
    except ms.ManifestError as e:
        errors.append(str(e)); study = None
    try:
        ms.validate_folds(ms.load_folds(), list_operators())
    except ms.ManifestError as e:
        errors.append(str(e))
    try:
        ms.load_arithmetic_modes()["default_by_dtype"]
    except Exception as e:  # noqa: BLE001
        errors.append(f"arithmetic_modes.yaml: {e}")
    from tilebench.llm.v2.skills.loader import SkillError, load_component, load_manifest
    try:
        m = load_manifest()
        for kind in ("reference", "device", "optimization"):
            for key, versions in m.get(kind, {}).items():
                for ver, e in versions.items():
                    if e.get("path"):
                        try:
                            load_component(m, kind, key, ver, require_status=(), provider_sendable=False)
                        except SkillError as ex:
                            errors.append(str(ex))
    except SkillError as e:
        errors.append(str(e))
    from tilebench.llm.v2.contracts.loader import audit_summary
    summ = audit_summary()
    for op, err in summ["invalid"].items():
        errors.append(f"contract {op}: {err}")
    missing = sorted(set(list_operators()) - set(summ["approved"] + summ["draft"] + summ["needs-review"] + list(summ["invalid"])))
    from tilebench.llm.v2.validation.contract_checks import RULE_SCOPES
    from tilebench.llm.v2.contracts.loader import load_contract
    for op in summ["approved"] + summ["draft"] + summ["needs-review"]:
        for r in load_contract(op, require_approved=False).rules.get("forbidden_substitutions", []):
            if r.get("scope", "host") not in RULE_SCOPES:
                errors.append(f"contract {op}: rule scope {r.get('scope')!r} invalid")
    _print({"errors": errors, "contracts": {k: (len(v) if isinstance(v, list) else len(v)) for k, v in summ.items()},
            "contracts_missing": missing, "study": "ok" if study else "invalid"})
    return 1 if errors else 0


def cmd_select_cases(a) -> int:
    from tilebench.llm.v2.tasks.case_selection import all_selections
    rows = [s.to_dict() for s in all_selections()]
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")
    _print([{k: r[k] for k in ("operator", "dtype", "case_index", "problem_size", "case_id", "params")} for r in rows])
    return 0


def cmd_tasks(a) -> int:
    from tilebench.llm.v2.tasks.support import summarize, task_table
    table = task_table(ms.load_study(), ms.load_folds())
    if a.out:
        Path(a.out).write_text(json.dumps([e.to_dict() for e in table], indent=1) + "\n")
    _print(summarize(table))
    return 0


def cmd_render(a) -> int:
    from tilebench.llm.v2.orchestration.campaign import build_task_context
    from tilebench.llm.v2.prompts.renderer import render_initial, render_system
    from tilebench.llm.v2.skills.loader import load_manifest
    from tilebench.llm.v2.tasks.support import eligibility
    study, folds = ms.load_study(), ms.load_folds()
    e = eligibility(a.operator, a.dtype, a.device, a.dsl, study, folds)
    ctx = build_task_context(e, study, load_manifest(), a.condition, require_approved=not a.allow_draft,
                             provider_sendable=False)
    system, user = render_system(ctx), render_initial(ctx)
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "system.md").write_text(system); (out / "user.md").write_text(user)
        print(f"wrote {out}/system.md ({len(system)} chars), user.md ({len(user)} chars)")
    else:
        print(system); print("\n" + "=" * 80 + "\n"); print(user)
    return 0


def cmd_dry_run(a) -> int:
    from tilebench.llm.v2.orchestration.campaign import dry_run
    rows = dry_run(a.device, a.dsl, a.condition, operators=a.operators, require_approved=not a.allow_draft, limit=a.limit)
    _print(rows)
    return 0 if not any("error" in r for r in rows) else 1


def cmd_preflight(a) -> int:
    from tilebench.llm.v2.orchestration.campaign import preflight
    reports = []
    study = ms.load_study()
    for device in (a.devices or list(study["support_matrix"])):
        for dsl in study["support_matrix"][device]["dsls"]:
            for cond in (a.conditions or ["base", "enhanced"]):
                reports.append(preflight(device, dsl, cond, live=a.live, study=study, run_type=a.run_type,
                                         provider=a.provider).to_dict())
    _print(reports)
    return 0 if all(r["ok"] for r in reports) else 1


# --------------------------------------------------------------------------
# live campaign commands
# --------------------------------------------------------------------------

def _campaign_spec(a, condition: str):
    from tilebench.llm.v2.orchestration.campaign import CampaignSpec
    return CampaignSpec(name=a.campaign, run_type=a.run_type, device=a.device, dsl=a.dsl, condition=condition,
                        model=a.model, operators=a.operators, dtypes=a.dtypes,
                        out_root=Path(a.out_root) if a.out_root else None, resume=a.resume,
                        stop_after_rounds=a.stop_after_rounds, isolation=a.isolation,
                        worker_timeout_s=a.worker_timeout, max_trajectories=a.max_trajectories,
                        retry_incomplete=a.retry_incomplete)


def _run_live(a, condition: str) -> int:
    from tilebench.llm.v2.orchestration.campaign import run_campaign
    spec = _campaign_spec(a, condition)
    summary = run_campaign(spec, log=_log)
    _print(summary)
    if summary.get("refused"):
        print(f"REFUSED: {len(summary['preflight']['blockers'])} blocker(s); no API call, no device run.", file=sys.stderr)
        return 2
    return 0


def cmd_base(a) -> int:
    return _run_live(a, "base")


def cmd_enhanced(a) -> int:
    return _run_live(a, "enhanced")


def cmd_probe_provider(a) -> int:
    """One real request with the configured model, effort and output cap
    (never a smaller model or a lower cap). Records request settings, the
    provider's echoed model id, terminal status and raw usage; the key is
    never printed."""
    from tilebench.llm.v2.providers.base import GenerationRequest, ProviderConfigError, TransportError
    from tilebench.llm.v2.providers.factory import build_provider, generator_spec
    models = ms.load_models()
    spec = generator_spec(models, a.model, accept_status=("approved", "candidate"))
    provider = build_provider(spec, timeout_s=float(models.get("transport", {}).get("timeout_s", 3600)))
    req = GenerationRequest(system="You are a terse assistant used for a connectivity check.",
                            user=a.prompt, model_id=spec.model_id, provider=spec.provider, settings=spec.settings,
                            metadata={"probe": True})
    t0 = time.time()
    record = {"generator": spec.record(), "prompt": a.prompt, "requested_at": t0}
    try:
        res = provider.generate(req)
    except ProviderConfigError as e:
        record.update({"outcome": "refused", **e.record()})
        _print(record)
        return 2
    except TransportError as e:
        record.update({"outcome": "transport_failed", **e.record()})
        _print(record)
        return 3
    record.update({"outcome": "ok", "response_id": res.response_id, "model_id_echoed": res.model_id,
                   "requested_model_id": res.requested_model_id, "terminal_status": res.terminal_status,
                   "truncated": res.truncated, "stream_events": res.stream_events, "streamed": res.streamed,
                   "usage_raw": res.usage_raw, "usage": res.usage.to_dict(), "elapsed_s": res.elapsed_s,
                   "request_settings_sent": res.request_settings_sent, "text": res.text, "error": res.error})
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({**record, "raw_response": res.raw_response}, indent=1, default=str) + "\n")
    _print(record)
    return 0 if res.text else 1


def cmd_distill(a) -> int:
    """Real distillation: trusted index -> scope -> map (one request per
    trajectory) -> reduce -> skill + manifest, persisted and resumable.
    Formal mode needs an approved distiller, frozen folds and formal Base
    data; test-only mode accepts a candidate distiller and writes a
    test-only skill that is never registered."""
    from tilebench.llm.v2.distillation.access import build_index, evaluation_scope, release_scope
    from tilebench.llm.v2.distillation.orchestrator import DistillerConfig, extract_observations, synthesize, write_skill
    from tilebench.llm.v2.providers.factory import build_provider, distiller_spec
    study, models, folds = ms.load_study(), ms.load_models(), ms.load_folds()
    formal = a.run_type == "formal"
    blockers = []
    if formal:
        blockers += ms.blockers_models(models, roles=("distiller",)) + ms.blockers_folds(folds)
    root = Path(a.campaign_dir)
    index = build_index(root)
    scope = release_scope(index, dsl=a.dsl, study=study, root=root) if a.mode == "release" else \
        evaluation_scope(index, dsl=a.dsl, held_out_fold=a.fold, study=study, root=root)
    if not scope.selected:
        blockers.append("no eligible source trajectories in the index (formal, base, source device, training folds)")
    report = {"scope": scope.manifest(), "blockers": blockers, "run_type": a.run_type, "index_size": len(index)}
    if blockers:
        _print(report)
        print("REFUSED: " + "; ".join(blockers), file=sys.stderr)
        return 2
    try:
        dspec = distiller_spec(models, accept_status=("approved",) if formal else ("approved", "candidate"))
    except ms.ManifestError as e:
        report["blockers"].append(str(e))
        _print(report)
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    provider = build_provider(dspec, timeout_s=float(models.get("transport", {}).get("timeout_s", 3600)))
    cfg = DistillerConfig(model_id=dspec.model_id, provider_name=dspec.provider, settings=dspec.settings,
                          dsl_version=study["dsls"][a.dsl]["reference_version"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scope.json").write_text(json.dumps({**report, "scope_sha256": scope.sha256(), "distiller": dspec.record()},
                                               indent=1, sort_keys=True) + "\n")
    obs = extract_observations(scope, provider, cfg, read_state=lambda p: json.loads(p.read_text()), out_dir=out, log=_log)
    res = synthesize(scope, obs, provider, cfg, models=sorted({r.model for r in scope.selected}), out_dir=out)
    write_skill(out / "skill", res, test_only=not formal)
    _print({"out": str(out), "observations": len(obs), "content_sha256": res["manifest"]["content_sha256"],
            "status": "draft" if formal else "test-only", "registered": False})
    return 0


def cmd_review_resolve(a) -> int:
    """Record a compliance decision for a review_required trajectory."""
    from tilebench.llm.v2.orchestration import state_machine as sm
    from tilebench.llm.v2.orchestration.state import TrajectoryState
    tpath = Path(a.trajectory_dir) / "trajectory.json"
    st = TrajectoryState.load(tpath)
    rec = st.rounds[-1]
    att = rec.attempts[-1]
    evidence = (att.compliance or {}).get("review_items", [])
    if a.show:
        _print({"trajectory_id": st.trajectory_id, "round": rec.round, "attempt": att.attempt, "review_items": evidence,
                "source_path": att.source_path})
        return 0
    if a.recheck:
        # Re-run the current checker on the stored candidate (e.g. after a checker fix). Only a `clear`
        # re-check resolves the review; anything else leaves the trajectory blocked for a human decision.
        from tilebench.llm.v2.contracts.loader import load_contract
        from tilebench.llm.v2.validation.contract_checks import CHECKER_VERSION, check_compliance
        rules = load_contract(st.task["operator"], require_approved=False).rules
        res = check_compliance(Path(att.source_path).read_text(), st.task["dsl"], rules)
        rc = res.to_dict(); rc["diagnostics"] = res.diagnostics(); rc["checker_version"] = CHECKER_VERSION
        (Path(a.trajectory_dir) / f"round_{rec.round:02d}" / f"attempt_{att.attempt}" / "compliance_recheck.json").write_text(
            json.dumps(rc, indent=1) + "\n")
        if res.verdict != "clear":
            _print({"trajectory_id": st.trajectory_id, "round": rec.round, "recheck_verdict": res.verdict,
                    "review_items": res.review_items(), "resolved": False})
            return 1
        a.decision = "compliant"
        a.note = (a.note or "") + f" [recheck under {CHECKER_VERSION}: clear; previous items: {evidence}]"
        a.reviewer = f"recheck:{CHECKER_VERSION}"
        att.compliance = rc
    if not a.decision:
        print("a --decision (or --recheck) is required", file=sys.stderr)
        return 2
    sm.resolve_review(st, rec.round, a.decision, a.note, reviewer=a.reviewer)
    st.save(tpath)
    from tilebench.llm.v2.providers.ledger import append_jsonl
    append_jsonl(Path(a.trajectory_dir) / "reviews.jsonl", {"round": rec.round, "attempt": att.attempt, "decision": a.decision,
                                                             "note": a.note, "reviewer": a.reviewer, "evidence": evidence,
                                                             "t": time.time()})
    _print({"trajectory_id": st.trajectory_id, "round": rec.round, "decision": a.decision, "status": st.status})
    return 0


def cmd_run_mock(a) -> int:
    """Ten-round mock execution with persistence, for smoke/resume demos."""
    from tilebench.llm.v2.devtools import run_mock_trajectory
    out = run_mock_trajectory(Path(a.out), operator=a.operator, dtype=a.dtype, device=a.device, dsl=a.dsl,
                             condition=a.condition, resume=a.resume)
    _print(out)
    return 0


def cmd_metrics(a) -> int:
    from tilebench.llm.v2.devtools import recompute_metrics
    _print(recompute_metrics(Path(a.campaign_dir), budgets=a.budgets))
    return 0


def cmd_campaign_report(a) -> int:
    from tilebench.llm.v2.devtools import campaign_report, campaign_report_markdown
    rep = campaign_report(Path(a.campaign_dir))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
        Path(a.out).with_suffix(".md").write_text(campaign_report_markdown(rep))
    print(campaign_report_markdown(rep))
    return 0


def cmd_coverage(a) -> int:
    from tilebench.llm.v2.devtools import coverage
    _print(coverage(Path(a.campaign_dir)))
    return 0


def cmd_review_queue(a) -> int:
    from tilebench.llm.v2.devtools import review_queue
    _print(review_queue(Path(a.campaign_dir)))
    return 0


def cmd_skills_register(a) -> int:
    from tilebench.llm.v2.skills.loader import MANIFEST_PATH, MANIFEST_SCHEMA, load_manifest, register_asset, save_manifest
    try:
        m = load_manifest()
    except Exception:  # noqa: BLE001
        m = {"schema": MANIFEST_SCHEMA}
    entry = register_asset(m, a.kind, a.key, a.version, a.path, permission=a.permission, status=a.status,
                           source=a.source, attachments=a.attachments or [], sendable_to=a.sendable_to or [],
                           publishable=a.publishable)
    save_manifest(m)
    _print({"registered": f"{a.kind}/{a.key}@{a.version}", "entry": entry, "manifest": str(MANIFEST_PATH)})
    return 0


def cmd_export_review_bundle(a) -> int:
    from tilebench.llm.v2.devtools import export_review_bundle
    _print(export_review_bundle(Path(a.out)))
    return 0


def cmd_export_publication(a) -> int:
    from tilebench.llm.v2.orchestration.publication import export_publication
    from tilebench.llm.v2.skills.loader import load_manifest
    _print(export_publication(Path(a.campaign_dir), Path(a.out), manifest=load_manifest()))
    return 0


def cmd_canonical_audit(a) -> int:
    from tilebench.llm.v2.devtools import canonical_audit_report
    text = canonical_audit_report()
    out = Path(a.out) if a.out else Path("docs/llm_v2/CANONICAL_AUDIT.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(text)} chars)")
    return 0


def cmd_snapshots(a) -> int:
    from tilebench.llm.v2.devtools import make_snapshots
    _print(make_snapshots(Path(a.out), operator=a.operator, dtype=a.dtype, device=a.device, dsl=a.dsl,
                          use_manifest=not a.synthetic))
    return 0


def cmd_capture_device(a) -> int:
    from tilebench.llm.v2.devtools import capture_device_facts
    _print(capture_device_facts())
    return 0


# --------------------------------------------------------------------------
def _add_live_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--device", required=True); s.add_argument("--dsl", required=True)
    s.add_argument("--model", required=True, help="generator role key in models.yaml (gpt | claude)")
    s.add_argument("--campaign", required=True, help="campaign name (output directory under --out-root)")
    s.add_argument("--run-type", default="formal", choices=list(ms.RUN_TYPES),
                   help="formal = scored (approved assets/models); validation = unscored engineering acceptance")
    s.add_argument("--operators", nargs="*"); s.add_argument("--dtypes", nargs="*")
    s.add_argument("--out-root", help="default outputs/llm_v2")
    s.add_argument("--resume", action="store_true", help="continue persisted trajectories (never re-requests archived responses)")
    s.add_argument("--stop-after-rounds", type=int, help="pause after this many closed rounds (resumable)")
    s.add_argument("--isolation", default="auto", choices=["auto", "bwrap", "none"])
    s.add_argument("--worker-timeout", type=int, default=1800)
    s.add_argument("--max-trajectories", type=int)
    s.add_argument("--retry-incomplete", action="store_true",
                   help="with --resume: re-evaluate the last round of a trajectory marked incomplete by an evaluation-side "
                        "infrastructure failure (no new request)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tilebench.llm.v2", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="environment, device, manifests, skills, contract status, isolation backend").set_defaults(fn=cmd_doctor)
    sub.add_parser("inventory", help="registered skill assets with hashes and grants").set_defaults(fn=cmd_inventory)
    sub.add_parser("validate-manifests", help="validate study/folds/modes/skills/contracts").set_defaults(fn=cmd_validate_manifests)
    s = sub.add_parser("select-cases", help="representative case per (operator, dtype)"); s.add_argument("--out"); s.set_defaults(fn=cmd_select_cases)
    s = sub.add_parser("tasks", help="task eligibility table"); s.add_argument("--out"); s.set_defaults(fn=cmd_tasks)

    s = sub.add_parser("render", help="render the initial prompt of one task")
    for name in ("--operator", "--dtype", "--device", "--dsl"):
        s.add_argument(name, required=True)
    s.add_argument("--condition", default="base", choices=["base", "enhanced"])
    s.add_argument("--allow-draft", action="store_true", help="accept draft skills/contracts (development only)")
    s.add_argument("--out"); s.set_defaults(fn=cmd_render)

    s = sub.add_parser("dry-run", help="render every eligible task of a device/DSL without any API call")
    s.add_argument("--device", required=True); s.add_argument("--dsl", required=True)
    s.add_argument("--condition", default="base", choices=["base", "enhanced"])
    s.add_argument("--operators", nargs="*"); s.add_argument("--limit", type=int)
    s.add_argument("--allow-draft", action="store_true"); s.set_defaults(fn=cmd_dry_run)

    s = sub.add_parser("preflight", help="per device/DSL/condition readiness report")
    s.add_argument("--devices", nargs="*"); s.add_argument("--conditions", nargs="*")
    s.add_argument("--live", action="store_true", help="apply live-campaign gates")
    s.add_argument("--run-type", default="formal", choices=list(ms.RUN_TYPES))
    s.add_argument("--provider", help="check provider grants for this provider (openai | anthropic)")
    s.set_defaults(fn=cmd_preflight)

    for name, fn in (("base", cmd_base), ("enhanced", cmd_enhanced)):
        s = sub.add_parser(name, help=f"{name} campaign: preflight, then the live chain (provider -> runner -> isolated evaluator)")
        _add_live_args(s); s.set_defaults(fn=fn)

    s = sub.add_parser("probe-provider", help="one real request with the configured model/effort/cap; records ids, status, usage")
    s.add_argument("--model", required=True); s.add_argument("--prompt", default="Reply with the single word OK.")
    s.add_argument("--out", help="write the full record (incl. raw response) to this JSON file"); s.set_defaults(fn=cmd_probe_provider)

    s = sub.add_parser("distill", help="source-only distillation (map/reduce, persisted, resumable); refuses without approved distiller/frozen folds in formal mode")
    s.add_argument("--campaign-dir", required=True, help="root holding the Base trajectories (trusted index is built from it)")
    s.add_argument("--dsl", required=True); s.add_argument("--fold", choices=["A", "B", "C"])
    s.add_argument("--mode", default="evaluation", choices=["evaluation", "release"])
    s.add_argument("--run-type", default="formal", choices=["formal", "test-only"])
    s.add_argument("--out", required=True); s.set_defaults(fn=cmd_distill)

    s = sub.add_parser("review-resolve", help="record a human compliance decision for a review_required trajectory")
    s.add_argument("--trajectory-dir", required=True)
    s.add_argument("--decision", choices=["compliant", "violation"])
    s.add_argument("--note", default=""); s.add_argument("--reviewer", default="human")
    s.add_argument("--show", action="store_true", help="print the pending evidence without deciding")
    s.add_argument("--recheck", action="store_true",
                   help="re-run the current checker on the stored candidate; resolves only if the re-check is clear")
    s.set_defaults(fn=cmd_review_resolve)

    s = sub.add_parser("run-mock", help="ten-round mock trajectory with persistence and resume")
    s.add_argument("--out", required=True)
    for name, default in (("--operator", "vector_add"), ("--dtype", "fp16"), ("--device", "B200"), ("--dsl", "triton")):
        s.add_argument(name, default=default)
    s.add_argument("--condition", default="base", choices=["base", "enhanced"])
    s.add_argument("--resume", action="store_true"); s.set_defaults(fn=cmd_run_mock)

    s = sub.add_parser("metrics", help="recompute E(B) curves from a campaign directory")
    s.add_argument("campaign_dir"); s.add_argument("--budgets", nargs="*", type=int); s.set_defaults(fn=cmd_metrics)
    s = sub.add_parser("coverage", help="task coverage / status counts of a campaign directory"); s.add_argument("campaign_dir"); s.set_defaults(fn=cmd_coverage)
    s = sub.add_parser("campaign-report", help="per-trajectory evidence table (ids, usage, timing, reviews) of a campaign directory")
    s.add_argument("campaign_dir"); s.add_argument("--out", help="write JSON here (and .md beside it)"); s.set_defaults(fn=cmd_campaign_report)
    s = sub.add_parser("review-queue", help="trajectories waiting for a compliance decision"); s.add_argument("campaign_dir"); s.set_defaults(fn=cmd_review_queue)

    s = sub.add_parser("skills-register", help="register/refresh one asset in skills/manifest.json")
    s.add_argument("--kind", required=True, choices=["reference", "device", "optimization", "contract"])
    s.add_argument("--key", required=True); s.add_argument("--version", required=True); s.add_argument("--path", required=True)
    s.add_argument("--permission", required=True, choices=["public", "internal", "private"])
    s.add_argument("--status", required=True, choices=["draft", "approved", "frozen", "test-only"])
    s.add_argument("--source", default=""); s.add_argument("--attachments", nargs="*")
    s.add_argument("--sendable-to", nargs="*", help="providers granted to receive this asset (openai anthropic)")
    s.add_argument("--publishable", action="store_true"); s.set_defaults(fn=cmd_skills_register)

    s = sub.add_parser("export-review-bundle", help="safe review bundle (no credentials, no private/non-publishable bodies)")
    s.add_argument("--out", required=True); s.set_defaults(fn=cmd_export_review_bundle)
    s = sub.add_parser("export-publication", help="Git-tracked publication copy of a campaign (every round; withholds non-publishable context)")
    s.add_argument("--campaign-dir", required=True); s.add_argument("--out", required=True); s.set_defaults(fn=cmd_export_publication)
    sub.add_parser("capture-device", help="print the runtime device facts a Device Context needs").set_defaults(fn=cmd_capture_device)
    s = sub.add_parser("canonical-audit", help="compile docs/llm_v2/CANONICAL_AUDIT.md from the 45 audit.json files")
    s.add_argument("--out"); s.set_defaults(fn=cmd_canonical_audit)
    s = sub.add_parser("snapshots", help="write representative rendered prompts with the real renderer")
    s.add_argument("--out", required=True)
    for name, default in (("--operator", "vector_add"), ("--dtype", "fp16"), ("--device", "B200"), ("--dsl", "triton")):
        s.add_argument(name, default=default)
    s.add_argument("--synthetic", action="store_true", help="use synthetic test-only components instead of the manifest")
    s.set_defaults(fn=cmd_snapshots)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
