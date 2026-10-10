"""Build the frozen Trn2 NKI LLM eligibility manifest (protocol revision 4).

    python scripts/llm_v2/build_trn2_eligibility.py [--check]

Rule (study owner, Trn2 NKI Base preparation instruction, 2026-10-10 §3, and launch decision T2): a task is
the operator's predeclared representative dtype (manifests/representative_dtypes.yaml) on its 20 frozen cases
(manifests/case_sets.yaml). It is eligible only when
  1. ALL 20 cases have a valid formal PyTorch eager baseline on the native Trn2 stack
     (results/TRN2/csv/<op>_default.csv: exactly one matching row per case, torch_status == "ok" and a finite
     positive torch_ms). Any CPU fallback, unresolved or failed baseline case excludes the operator; and
  2. ALL 20 cases have a supported empirical SOL target: metrics.empirical.t_emp with the declared mode
     (manifests/arithmetic_modes.yaml) and the Trn2 profile registered in manifests/calibration.yaml gives status
     "ok". A case without a calibrated ceiling (e.g. peak_unavailable) excludes the operator as
     scoring_<status>; no other mode or theoretical peak is substituted.
The dtype is never changed to avoid an exclusion. Nothing is re-run here.

Case matching uses the framework's own CSV label matcher (metrics.baselines.label_matches), the same one that
builds the stored torch baselines. Writes tilebench/llm/v2/manifests/trn2_eligibility.json (sealed);
--check rebuilds in memory and compares with the committed file."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tilebench.llm.v2.manifests.schema import canonical_json, sha256_text  # noqa: E402
from tilebench.llm.v2.metrics import empirical  # noqa: E402
from tilebench.llm.v2.metrics.baselines import label_matches  # noqa: E402
from tilebench.llm.v2.tasks import case_sets as cs_mod  # noqa: E402
from tilebench.llm.v2.tasks.case_selection import load_operator_config  # noqa: E402
from tilebench.llm.v2.tasks.representative import selected  # noqa: E402

OUT = REPO / "tilebench" / "llm" / "v2" / "manifests" / "trn2_eligibility.json"
CSV_ROOT = REPO / "results" / "TRN2" / "csv"
SCHEMA = "tilebench-llm-v2-trn2-eligibility/2"
# Provenance of results/TRN2/csv: the formal native Trn2 results merged on main (#322, 33e40ace; main 72d7cec6),
# brought to exp/llm-trn2 by a plain file checkout (byte-identical to main).
SOURCE_COMMIT = {"results_pr": "#322", "results_commit": "33e40ace", "main_commit": "72d7cec6"}

# Operator-level explanation of an exclusion (evidence: the CSV rows; for whole-op CPU fallbacks also the
# fallback classification of 2026-10-06, torch_neuronx.get_fallback_ops()).
EXPLANATION = {
    "argmax": "PyTorch eager baseline is a whole-op CPU fallback on the native Neuron stack (aten::argmax.out, "
              "0 device executions)",
    "2d_max_pooling": "PyTorch eager baseline is a whole-op CPU fallback on the native Neuron stack "
                      "(aten::max_pool2d_with_indices.out, 0 device executions)",
    "radix_sort": "PyTorch eager baseline (torch.sort) is a whole-op CPU fallback on the native Neuron stack "
                  "(aten::sort.values_stable); baseline unresolved",
    "flash_attention": "PyTorch eager baseline (scaled_dot_product_attention, fp16) fails the operator's verify "
                       "tolerance on the device (baseline correctness failure); baseline unresolved",
    "block_sparse_attention": "the PyTorch baseline is flex_attention, which cannot run without torch.compile; "
                              "no pure eager baseline (unresolved)",
    "gaussian_blur": "a selected fp16 case has no valid PyTorch eager baseline (compiler/runtime failure)",
    "reverse_array": "selected fp16 cases have no valid PyTorch eager baseline (compiler/runtime failure)",
    "top_k_selection": "a selected fp32 case has no valid PyTorch eager baseline (compiler/runtime failure)",
    "matmul_int8": "runnable on the device (20/20 valid formal baselines; not a CPU fallback) but no supported SOL "
                   "scoring mode: the declared mode mma_int8_i32acc has no calibrated Trn2 ceiling (NeuronCore-v3 "
                   "TensorE has no int8 input type); no bf16 or theoretical peak is substituted",
}
# Exclusion class of a baseline failure (rule 1); a scoring exclusion (rule 2) is scoring_<t_emp status>.
BASELINE_CLASS = {
    "argmax": "cpu_fallback", "2d_max_pooling": "cpu_fallback", "radix_sort": "cpu_fallback",
    "flash_attention": "baseline_correctness_failure", "block_sparse_attention": "no_eager_baseline",
    "gaussian_blur": "baseline_case_failure", "reverse_array": "baseline_case_failure",
    "top_k_selection": "baseline_case_failure",
}
DECISIONS = {
    "T2_scoring_availability": "study owner, Trn2 NKI Base launch decision T2 (2026-10-10): every included task needs "
                               "valid formal baseline coverage AND a supported SOL scoring mode; matmul_int8 is "
                               "excluded for scoring_peak_unavailable (not CPU fallback)",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_evidence(rows: list[dict], case: dict) -> dict:
    hits = [r for r in rows if label_matches(r["params"], case["params"], case.get("problem_size"))]
    ev = {"case_id": case["case_id"], "case_index": case["case_index"], "params": case["params"],
          "matching_rows": len(hits)}
    if len(hits) != 1:
        ev.update(baseline_valid=False, problem=f"{len(hits)} matching CSV rows (exactly one required)")
        return ev
    r = hits[0]
    try:
        t = float(r["torch_ms"])
    except (TypeError, ValueError):
        t = float("nan")
    valid = r.get("torch_status") == "ok" and t > 0 and t == t
    ev.update(csv_params=r["params"], torch_status=r.get("torch_status"),
              torch_ms=t if t == t else None, nki_status=r.get("nki_status"),
              na_scope=r.get("na_scope") or None, na_reason=r.get("na_reason") or None, baseline_valid=valid)
    return ev


def _scoring(op: str, dtype: str, cases: list[dict], profile: dict, modes_doc: dict) -> dict:
    """Rule 2: the empirical target of every frozen case (the scoring-table computation)."""
    metrics_cfg = load_operator_config(op).get("metrics", {})
    recs = [empirical.t_emp("Trn2", op, dtype, c["params"], c.get("problem_size", 1), metrics_cfg, modes_doc, profile)
            for c in cases]
    counts: dict[str, int] = {}
    for r in recs:
        counts[r.status] = counts.get(r.status, 0) + 1
    bad = [r for r in recs if r.status != "ok"]
    return {"mode": recs[0].mode if recs else None, "status_counts": counts, "supported": not bad and len(recs) == 20,
            "problem_status": bad[0].status if bad else None, "problem": bad[0].reason if bad else None}


def build() -> dict:
    sets = cs_mod.load_manifest()["operators"]
    entry_cal = empirical.device_entry("Trn2")
    profile = empirical.load_profile(entry_cal)          # sha256-verified against calibration.yaml
    modes_doc = empirical.load_modes()
    chosen = selected()
    ops, counts = {}, {"eligible": 0, "excluded": 0}
    for op in sorted(sets):
        entry = sets[op]
        dtype = entry["dtype"]
        if chosen.get(op) != dtype:
            raise SystemExit(f"{op}: case-set dtype {dtype} != representative dtype {chosen.get(op)}")
        path = CSV_ROOT / f"{op}_default.csv"
        rows = [r for r in csv.DictReader(path.open(newline="")) if r.get("dtype") == dtype] if path.exists() else []
        cases = [_case_evidence(rows, c) for c in entry["cases"]]
        n_valid = sum(1 for c in cases if c["baseline_valid"])
        n = len(entry["cases"])
        baseline_ok = n == 20 and n_valid == n
        scoring = _scoring(op, dtype, entry["cases"], profile, modes_doc)
        eligible = baseline_ok and scoring["supported"]
        bad = [c for c in cases if not c["baseline_valid"]]
        reasons = sorted({c.get("na_reason") or c.get("problem") or c.get("torch_status") or "invalid" for c in bad})
        if eligible:
            exclusion_class = None
            reason = (f"20/20 configured {dtype} cases have a valid formal PyTorch eager baseline on the native Trn2 "
                      f"stack and a supported empirical SOL target (mode {scoring['mode']})")
        elif not baseline_ok:
            exclusion_class = BASELINE_CLASS.get(op, "baseline_invalid")
            reason = (f"{EXPLANATION.get(op, 'baseline not valid on every case')}: {n_valid}/{n} configured {dtype} "
                      f"cases have a valid baseline; CSV reasons {reasons}")
        else:
            exclusion_class = f"scoring_{scoring['problem_status']}"
            reason = (f"{EXPLANATION.get(op, 'no supported SOL scoring mode')}: {n_valid}/{n} configured {dtype} cases "
                      f"have a valid baseline, empirical targets {scoring['status_counts']} ({scoring['problem']})")
        counts["eligible" if eligible else "excluded"] += 1
        ops[op] = {"dtype": dtype, "case_set_id": entry["case_set_id"], "n_cases": n,
                   "cases_matched": sum(1 for c in cases if c["matching_rows"] == 1), "baseline_valid_cases": n_valid,
                   "baseline_complete": baseline_ok, "scoring": scoring,
                   "status": "eligible" if eligible else "excluded", "exclusion_class": exclusion_class, "reason": reason,
                   "csv": {"path": str(path.relative_to(REPO)), "sha256": _sha256(path) if path.exists() else None,
                           "rows_for_dtype": len(rows)},
                   "cases": cases}
    body = {
        "schema": SCHEMA, "device": "Trn2", "dsl": "nki", "status": "frozen", "frozen_on": "2026-10-10",
        "approved_by": "study owner, Trn2 NKI Base preparation instruction (2026-10-10) §3: an operator whose selected "
                       "task has any CPU-fallback or otherwise invalid formal baseline case is excluded; 20/20 valid "
                       "baseline cases are required; the dtype is never changed. Launch decision T2 (2026-10-10): "
                       "an included task also needs a supported SOL scoring mode on every case",
        "decisions": DECISIONS,
        "rule": "eligible iff all 20 frozen cases of the representative dtype (1) have exactly one row in "
                "results/TRN2/csv/<op>_default.csv with torch_status == 'ok' and finite positive torch_ms and "
                "(2) have empirical-target status 'ok' under the declared mode and the registered Trn2 profile",
        "scoring_source": {"calibration_id": profile.get("calibration_id"), "profile": entry_cal.get("profile"),
                           "profile_file_sha256": entry_cal.get("sha256"),
                           "profile_seal_sha256": profile.get("profile_sha256"),
                           "arithmetic_modes_sha256": empirical.modes_sha256(),
                           "ceiling_basis": profile.get("ceiling_basis")},
        "protocol": {"revision": 4, "operators": len(ops), "cases_per_task": 20,
                     "representative_dtypes_sha256": _sha256(REPO / "tilebench/llm/v2/manifests/representative_dtypes.yaml"),
                     "case_sets_sha256": _sha256(REPO / "tilebench/llm/v2/manifests/case_sets.yaml")},
        "source": {**SOURCE_COMMIT, "csv_root": "results/TRN2/csv",
                   "stack": "native PyTorch eager on torch.device('neuron') vs NKI 0.7, device busy sum, warmup 1 / repeat 3"},
        "counts": counts,
        "excluded": sorted(op for op, v in ops.items() if v["status"] == "excluded"),
        "excluded_by_class": {k: sorted(op for op, v in ops.items() if v["exclusion_class"] == k)
                              for k in sorted({v["exclusion_class"] for v in ops.values() if v["exclusion_class"]})},
        "operators": ops,
    }
    body["sha256"] = sha256_text(canonical_json({k: v for k, v in body.items() if k != "sha256"}))
    return body


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare a rebuild with the committed file")
    a = ap.parse_args(argv)
    body = build()
    text = json.dumps(body, indent=1, sort_keys=True) + "\n"
    if a.check:
        same = OUT.exists() and OUT.read_text() == text
        print("trn2_eligibility.json", "matches rebuild" if same else "DIFFERS from rebuild")
        return 0 if same else 1
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO)}: {body['counts']}, excluded {body['excluded']}, seal {body['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
