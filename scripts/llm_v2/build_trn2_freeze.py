"""Build the Trn2 NKI Base campaign freeze (protocol revision 4) from the tree.

    python scripts/llm_v2/build_trn2_freeze.py            # write tilebench/llm/v2/manifests/frozen/Trn2_base_rev4_2026-10-10.json
    python scripts/llm_v2/build_trn2_freeze.py --check    # rebuild in memory and compare (tests/llm_v2/test_frozen_trn2.py)

Same structure as frozen/B200_base_rev4_2026-10-07.json (schema tilebench-llm-v2-frozen-study/2) for the Trn2 / NKI
track: the eligible tasks of manifests/trn2_eligibility.json (36 after launch decision T2), Reference Skill
nki@0.7.0b1, Device Context Trn2@2026-10-10, native Neuron evaluator (timing adapter neuron_native_trace, docker
isolation), the Trn2 empirical profile, scoring table and torch baseline. `open_owner_decisions` lists what a formal
launch still needs; the owner's decisions are recorded by re-running with --decided ID=record ... (never edited by
hand). With T1/T2/T3 decided and the tree consistent, freeze_status is `frozen`."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tilebench.llm.v2.evaluation import fingerprint as fpm  # noqa: E402
from tilebench.llm.v2.evaluation import neuron as N  # noqa: E402
from tilebench.llm.v2.evaluation.worker import SUITE_STOP_STATUSES  # noqa: E402
from tilebench.llm.v2.manifests import schema as ms  # noqa: E402
from tilebench.llm.v2.metrics import baselines, cost  # noqa: E402
from tilebench.llm.v2.metrics import empirical as E  # noqa: E402
from tilebench.llm.v2.orchestration.campaign import protocol_identity  # noqa: E402
from tilebench.llm.v2.prompts.renderer import templates_sha256  # noqa: E402
from tilebench.llm.v2.skills.loader import load_manifest  # noqa: E402
from tilebench.llm.v2.tasks import representative, support  # noqa: E402

OUT = ms.MANIFEST_DIR / "frozen" / "Trn2_base_rev4_2026-10-10.json"
SCORING_TABLE = REPO / "artifacts/llm_v2/scoring/Trn2/scoring_table_Trn2_rev4_casesets_2026-10-10.json"
CAMPAIGN = "formal_trn2_nki_base_rev4_2026-10-10"
B200_FREEZE = ms.MANIFEST_DIR / "frozen" / "B200_base_rev4_2026-10-07.json"

SHARED_MANIFEST_DELTA = {
    "study.yaml": [
        "support_matrix.Trn2.timing_adapter: neuron_runtime_trace (legacy torch-xla, never ready) -> neuron_native_trace",
        "dsls.nki.reference_version: beta5 (private, metadata only) -> 0.7.0b1 (public Reference Skill)",
        "device_snapshots.Trn2: (absent) -> 2026-10-10",
        "torch_baselines.Trn2: (absent) -> artifacts/llm_v2/baselines/Trn2/torch_baseline_Trn2_casesets_2026-10-10.json",
    ],
    "calibration.yaml": ["devices.Trn2: none -> Trn2-20261010T012011Z-94406dcb (status as registered; frozen by launch "
                         "decision T1)"],
    "skills/manifest.json": ["reference.nki.0.7.0b1 added", "device.Trn2.2026-10-10 added; device.Trn2.2026-10-05 superseded_by"],
    "note": ("every other entry is unchanged; tests/llm_v2/test_frozen_rev4.py checks the B200 freeze on the B200 view of the "
             "study (exactly these Trn2 keys reverted). The study config hash of a Trn2 campaign therefore differs from the "
             "B200 one: these changes must not be merged into exp/llm-b200, exp/llm-gh200 or exp/llm-mi300x while their "
             "campaigns run."),
}


# Shared-definition scoring notes checked before launch (unchanged here; any correction is a central decision for every
# device, because arithmetic_modes.yaml is pinned by the running campaigns' scoring bindings).
SCORING_CAVEATS = {
    "rmsnorm": ("frozen Q = 3*n*dtype_size (two sweeps of x + one write; audit.json) is exactly 1.5x the compulsory "
                "(2*n + K)*dtype_size on all 20 Trn2 cases; memory-bound under both; uniform per-operator scale, so "
                "within-operator comparisons are unchanged; efficiency > 1 is possible and kept as an audit flag (never "
                "clipped); same frozen definition as B200/GH200/MI300X"),
    "linear_self_attention": ("fp32 task declared mma_tf32_f32acc (contract allows TF32 operands for S and the numerator; "
                              "the installed nc_matmul lists tfloat32 as a TensorE input type); all 20 Trn2 cases are memory-bound "
                              "(T_emp = Q / BW_emp), so the TF32 compute rate never sets the target"),
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(decided: dict | None = None) -> dict:
    study, models, folds, modes = ms.load_study(), ms.load_models(), ms.load_folds(), ms.load_arithmetic_modes()
    b200 = json.loads(B200_FREEZE.read_text())
    elig = support.load_trn2_eligibility()
    eligible = sorted(op for op, e in elig["operators"].items() if e["status"] == "eligible")
    cs = ms.load_case_sets()
    man = load_manifest()
    ref_ver = study["dsls"]["nki"]["reference_version"]
    dev_ver = study["device_snapshots"]["Trn2"]
    skills = {}
    for kind, key, ver in (("reference", "nki", ref_ver), ("device", "Trn2", dev_ver)):
        e = man[kind][key][ver]
        skills[f"{kind}/{key}@{ver}"] = {k: e.get(k) for k in ("asset_manifest", "permission", "sendable_to", "sha256_raw",
                                                               "sha256_injected", "status", "chars", "license")}
    skills["optimization"] = "none (Base)"
    entry = E.device_entry("Trn2")
    profile = E.load_profile(entry)
    table = json.loads(SCORING_TABLE.read_text())
    bl_path = REPO / study["torch_baselines"]["Trn2"]
    bl = baselines.load(bl_path)
    g = models["roles"]["generator"]
    job = fpm  # noqa: F841  (sources below)
    open_decisions = []
    if entry.get("status") != "frozen":
        open_decisions.append({"id": "T1", "decision": "freeze the Trn2 empirical profile Trn2-20261010T012011Z-94406dcb "
                               "(calibration-register --status frozen --by <owner>)", "blocks": "formal preflight"})
    unavailable = sorted({r["operator"] for r in table["rows"] if r["status"] != "ok"})
    scoring_excluded = elig.get("excluded_by_class", {}).get("scoring_peak_unavailable", [])
    if unavailable or not scoring_excluded:
        open_decisions.append({"id": "T2", "decision": f"scoring of execution-eligible tasks without an empirical ceiling: "
                               f"{unavailable or ['matmul_int8']} (mode mma_int8_i32acc unsupported on NeuronCore-v3); keep them as "
                               "execution-only tasks (no efficiency) or exclude them; never a substituted peak",
                               "blocks": "formal preflight (scoring_gate requires every task ok)"})
    open_decisions.append({"id": "T3", "decision": "accept the docker isolation backend for formal Trn2 runs (same allowlist "
                           "and probe as bwrap; bwrap cannot create user namespaces on this host without root)",
                           "blocks": "nothing in code (formal_isolation_backends(Trn2) accepts docker); owner sign-off"})
    for d in (decided or {}):
        open_decisions = [o for o in open_decisions if o["id"] != d]
    n_cases = sum(elig["operators"][op]["n_cases"] for op in eligible)
    ready = not open_decisions and not unavailable and table["status_counts"] == {"ok": n_cases}
    body = {
        "schema": "tilebench-llm-v2-frozen-study/2",
        "device": "Trn2", "dsls": ["nki"], "condition": "base", "models": ["gpt", "claude"],
        "campaign": CAMPAIGN,
        "S_llm": "S_llm-rev4-20case-2026-10-07 (protocol revision 4) + the Trn2 device additions on exp/llm-trn2 "
                 "(shared_manifest_delta); campaign runs from exp/llm-trn2",
        "shared_manifest_delta": SHARED_MANIFEST_DELTA,
        "protocol": {**protocol_identity(study), "study_yaml_sha256": _sha(ms.MANIFEST_DIR / "study.yaml"),
                     "task_unit_spec": study.get("task_unit")},
        "study_config_hash": ms.study_config_hash(study, models, folds, modes),
        "manifests": {p.name: _sha(p) for p in sorted(ms.MANIFEST_DIR.glob("*.yaml")) + [support.TRN2_ELIGIBILITY_PATH]},
        "case_sets": {"sha256": _sha(ms.MANIFEST_DIR / "case_sets.yaml"),
                      "case_set_ids": {op: cs["operators"][op]["case_set_id"] for op in eligible}},
        "representative_dtypes": {"mapping": {op: representative.selected()[op] for op in eligible}, "tasks_per_dsl": len(eligible)},
        "trn2_eligibility": {"path": str(support.TRN2_ELIGIBILITY_PATH.relative_to(REPO)), "seal": elig["sha256"],
                             "counts": elig["counts"], "excluded": elig["excluded"],
                             "excluded_by_class": elig.get("excluded_by_class"), "eligible": eligible},
        "campaign_size": {"operators": len(eligible), "cases_per_task": 20, "cases": n_cases, "models": 2,
                          "trajectories": len(eligible) * 2, "rounds_per_trajectory": study["trajectory"]["rounds"],
                          "generations_per_round": study["trajectory"]["max_generations_per_round"],
                          "generation_rounds": len(eligible) * 2 * study["trajectory"]["rounds"]},
        "checker": fpm.checker_fingerprint(),
        "input_kinds": {"file": "tilebench/llm/v2/manifests/input_kinds.yaml", "sha256": _sha(ms.MANIFEST_DIR / "input_kinds.yaml")},
        "contracts": {op: {"contract_md": _sha(REPO / "tilebench/llm/v2/contracts/data" / op / "contract.md"),
                           "evaluator_rules_json": _sha(REPO / "tilebench/llm/v2/contracts/data" / op / "evaluator_rules.json"),
                           "audit_json": _sha(REPO / "tilebench/llm/v2/contracts/data" / op / "audit.json")} for op in eligible},
        "contracts_identical_to_b200_freeze": all(
            (b200["contracts"][op]["contract_md"], b200["contracts"][op]["evaluator_rules_json"], b200["contracts"][op]["audit_json"]) ==
            (_sha(REPO / "tilebench/llm/v2/contracts/data" / op / "contract.md"),
             _sha(REPO / "tilebench/llm/v2/contracts/data" / op / "evaluator_rules.json"),
             _sha(REPO / "tilebench/llm/v2/contracts/data" / op / "audit.json")) for op in eligible),
        "skills": skills,
        "evaluator_fingerprint_policy": {
            "evaluation_sources_sha256": fpm._sources_sha256(fpm.V2_ROOT, fpm.EVALUATION_SOURCES),
            "core_sources_sha256": fpm._sources_sha256(fpm.CORE_ROOT, fpm.CORE_SOURCES),
            "neuron_evaluation_sources_sha256": fpm._sources_sha256(fpm.V2_ROOT, N.NEURON_EVALUATION_SOURCES),
            "neuron_core_sources_sha256": fpm._sources_sha256(fpm.CORE_ROOT, N.NEURON_CORE_SOURCES),
            "timing_adapter": N.ADAPTER, "isolation_backend": "docker", "worker_timeout_s": study["evaluation"]["worker_timeout_s"],
            "suite": {"cases_per_candidate": 20, "round_latency": "geometric_mean_of_case_means", "round_validity": "all_cases_valid",
                      "stop_suite_on": list(SUITE_STOP_STATUSES)},
            "timing": {"warmup": study["timing"]["warmup"], "repeat": study["timing"]["repeat"], "mode": "eager (no CUDA graph on Neuron)",
                       "measured_quantity": "device busy sum of every Neuron execution of one run() call (per-core copies merged)",
                       "flush": "none (Neuron runtime: inputs in HBM, SBUF/PSUM software-managed)"},
            "reference": "impl_torch.run on CPU copies of the device inputs (formal Trn2 definition)",
            "host_fallback": "torch_neuronx op tracking around every candidate call: CPU fallback op or neuron_to_cpu copy = contract_violation",
            "rule": "a formal resume refuses any evaluator-fingerprint difference (job, rules, checker, evaluation, neuron and core "
                    "sources, worker timeout, isolation backend, environment incl. torch / torch-neuronx / nki / neuronx-cc)"},
        "calibration_profile": {"calibration_id": profile.get("calibration_id"), "path": entry.get("profile"),
                                "file_sha256": entry.get("sha256"), "embedded_seal": profile.get("seal") or profile.get("profile_sha256"),
                                "status": entry.get("status")},
        "scoring_binding": E.scoring_binding("Trn2"),
        "scoring_table": {"path": str(SCORING_TABLE.relative_to(REPO)), "scoring_sha256": table["scoring_sha256"],
                          "status_counts": table["status_counts"], "ceiling_basis": table.get("ceiling_basis"),
                          "execution_eligible_scoring_unavailable": unavailable},
        "torch_baseline": {"path": str(bl_path.relative_to(REPO)), "sha256": bl["sha256"],
                           "cases_matched": sum(1 for op in eligible for v in bl["operators"][op]["cases"].values() if "torch_ms" in v)},
        "generators": {m: {k: g[m].get(k) for k in ("provider", "model_id", "reasoning_effort", "output_effort", "thinking",
                                                    "max_output_tokens", "status")} for m in ("gpt", "claude")},
        "scoring_caveats": SCORING_CAVEATS,
        "folds": b200["folds"], "arithmetic_modes": {"revision": modes.get("revision"), "sha256": E.modes_sha256(), "status": modes.get("status")},
        "pricing": {"file": "tilebench/llm/v2/manifests/api_pricing.yaml", "pricing_snapshot_sha256": cost.pricing_sha256()},
        "transport": models.get("transport"),
        "templates_sha256": templates_sha256(),
        "excluded_campaigns_sha256": _sha(ms.MANIFEST_DIR / "excluded_campaigns.yaml"),
        "compliance_review": b200["compliance_review"],
        "scheduling_policy": {**b200["scheduling_policy"], "gpu_evaluation": "serialized by the Trn2 device lock (one candidate at a time "
                                                                             "on one logical NeuronCore)",
                              "process_layout": "one `schedule` process for both models (--models gpt claude --dsls nki)"},
        "campaign_environment": "trn2.3xlarge; /home/ubuntu/workspace/native_venv (nki 0.7.0b1, neuronx-cc 2.0.404830.0a0, torch "
                                "2.14.0+cpu, torch-neuronx 2.14.3); docker sandbox (image tilebench-sandbox-empty:1); "
                                "NEURON_LOGICAL_NC_CONFIG=2, NEURON_RT_NUM_CORES=1; run under a memory-capped systemd user unit",
        "nominal_generations": len(eligible) * 2 * study["trajectory"]["rounds"],
        "freeze_status": "frozen" if ready else "prepared",
        "open_owner_decisions": open_decisions,
        "decided": decided or {},
        "frozen_by": ("study owner, Trn2 NKI Base launch decisions T1/T2/T3 (2026-10-10); integrated by the Claude Code "
                      "session") if ready else None,
        "frozen_on": "2026-10-10" if ready else None,
    }
    return body


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--decided", nargs="*", default=[], help="ID=record pairs of owner decisions (e.g. T3='owner, 2026-10-10')")
    a = ap.parse_args(argv)
    decided = dict(x.split("=", 1) for x in a.decided)
    if a.check:
        prev = json.loads(OUT.read_text())
        body = build(prev.get("decided") or {})
        same = json.dumps(body, sort_keys=True) == json.dumps(prev, sort_keys=True)
        print("Trn2 freeze", "matches the tree" if same else "DIFFERS from the tree")
        return 0 if same else 1
    body = build(decided)
    OUT.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}: freeze_status={body['freeze_status']}, tasks={body['representative_dtypes']['tasks_per_dsl']}, "
          f"open={[o['id'] for o in body['open_owner_decisions']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
