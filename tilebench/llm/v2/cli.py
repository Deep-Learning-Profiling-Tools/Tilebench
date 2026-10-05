"""Turnkey CLI: python -m tilebench.llm.v2 <command> [...]

Every live campaign command runs the preflight first and refuses to start
on any blocker. No command installs packages, edits settings or falls back
to another device's configuration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tilebench.paths import list_operators

from tilebench.llm.v2 import PROTOCOL
from tilebench.llm.v2.manifests import schema as ms


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, sort_keys=True, default=str))


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
        out["manifests"]["model_blockers"] = ms.blockers_models(ms.load_models())
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
                             "permission": e.get("permission"), "sha256_injected": (e.get("sha256_injected") or "")[:16],
                             "chars": e.get("chars")})
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
    ctx = build_task_context(e, study, load_manifest(), a.condition, require_approved=not a.allow_draft)
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
                reports.append(preflight(device, dsl, cond, live=a.live, study=study).to_dict())
    _print(reports)
    return 0 if all(r["ok"] for r in reports) else 1


def _campaign_gate(a, condition: str) -> int:
    from tilebench.llm.v2.orchestration.campaign import preflight
    pf = preflight(a.device, a.dsl, condition, live=True)
    _print(pf.to_dict())
    if not pf.ok:
        print(f"REFUSED: {len(pf.blockers)} blocker(s); no API call, no device run.", file=sys.stderr)
        return 2
    print("Preflight passed. Live execution is not wired to this command in the shared-framework phase; "
          "device branches add the runner invocation after S_llm is frozen.", file=sys.stderr)
    return 0


def cmd_base(a) -> int:
    return _campaign_gate(a, "base")


def cmd_enhanced(a) -> int:
    return _campaign_gate(a, "enhanced")


def cmd_distill(a) -> int:
    from tilebench.llm.v2.distillation.access import evaluation_scope, release_scope, refs_from_states
    study = ms.load_study()
    blockers = ms.blockers_models(ms.load_models(), roles=("distiller",)) + ms.blockers_folds(ms.load_folds())
    index = []
    if a.index:
        index = refs_from_states(json.loads(Path(a.index).read_text()))
    scope = release_scope(index, dsl=a.dsl, study=study) if a.mode == "release" else \
        evaluation_scope(index, dsl=a.dsl, held_out_fold=a.fold, study=study)
    _print({"scope": scope.manifest(), "blockers": blockers})
    if blockers:
        print("REFUSED: distillation needs an approved distiller model and frozen folds.", file=sys.stderr)
        return 2
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
                           source=a.source, attachments=a.attachments or [])
    save_manifest(m)
    _print({"registered": f"{a.kind}/{a.key}@{a.version}", "entry": entry, "manifest": str(MANIFEST_PATH)})
    return 0


def cmd_export_review_bundle(a) -> int:
    from tilebench.llm.v2.devtools import export_review_bundle
    _print(export_review_bundle(Path(a.out)))
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
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tilebench.llm.v2", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="environment, device, manifests, skills and contract status").set_defaults(fn=cmd_doctor)
    sub.add_parser("inventory", help="registered skill assets with hashes").set_defaults(fn=cmd_inventory)
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
    s.add_argument("--live", action="store_true", help="apply live-campaign gates (approved assets, model ids)")
    s.set_defaults(fn=cmd_preflight)

    for name, fn in (("base", cmd_base), ("enhanced", cmd_enhanced)):
        s = sub.add_parser(name, help=f"{name} campaign gate (preflight; refuses on blockers)")
        s.add_argument("--device", required=True); s.add_argument("--dsl", required=True); s.set_defaults(fn=fn)

    s = sub.add_parser("distill", help="distillation scope + gate (no API call here)")
    s.add_argument("--dsl", required=True); s.add_argument("--fold", choices=["A", "B", "C"])
    s.add_argument("--mode", default="evaluation", choices=["evaluation", "release"])
    s.add_argument("--index", help="JSON list of trajectory states (with 'path')"); s.set_defaults(fn=cmd_distill)

    s = sub.add_parser("run-mock", help="ten-round mock trajectory with persistence and resume")
    s.add_argument("--out", required=True)
    for name, default in (("--operator", "vector_add"), ("--dtype", "fp16"), ("--device", "B200"), ("--dsl", "triton")):
        s.add_argument(name, default=default)
    s.add_argument("--condition", default="base", choices=["base", "enhanced"])
    s.add_argument("--resume", action="store_true"); s.set_defaults(fn=cmd_run_mock)

    s = sub.add_parser("metrics", help="recompute E(B) curves from a campaign directory")
    s.add_argument("campaign_dir"); s.add_argument("--budgets", nargs="*", type=int); s.set_defaults(fn=cmd_metrics)
    s = sub.add_parser("coverage", help="task coverage / status counts of a campaign directory"); s.add_argument("campaign_dir"); s.set_defaults(fn=cmd_coverage)
    s = sub.add_parser("review-queue", help="trajectories waiting for a compliance decision"); s.add_argument("campaign_dir"); s.set_defaults(fn=cmd_review_queue)

    s = sub.add_parser("skills-register", help="register/refresh one asset in skills/manifest.json")
    s.add_argument("--kind", required=True, choices=["reference", "device", "optimization", "contract"])
    s.add_argument("--key", required=True); s.add_argument("--version", required=True); s.add_argument("--path", required=True)
    s.add_argument("--permission", required=True, choices=["public", "internal", "private"])
    s.add_argument("--status", required=True, choices=["draft", "approved", "frozen", "test-only"])
    s.add_argument("--source", default=""); s.add_argument("--attachments", nargs="*"); s.set_defaults(fn=cmd_skills_register)

    s = sub.add_parser("export-review-bundle", help="safe review bundle (no credentials, no private bodies)")
    s.add_argument("--out", required=True); s.set_defaults(fn=cmd_export_review_bundle)
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
