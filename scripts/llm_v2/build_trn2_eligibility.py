"""Build the frozen Trn2 NKI LLM eligibility manifest (protocol revision 4).

    python scripts/llm_v2/build_trn2_eligibility.py [--check]

Rule (study owner, Trn2 NKI Base preparation instruction, 2026-10-10 §3): a task is the operator's
predeclared representative dtype (manifests/representative_dtypes.yaml) on its 20 frozen cases
(manifests/case_sets.yaml). It is eligible only when ALL 20 cases have a valid formal PyTorch eager
baseline on the native Trn2 stack (results/TRN2/csv/<op>_default.csv: exactly one matching row per case,
torch_status == "ok" and a finite positive torch_ms). Any CPU fallback, unresolved or failed baseline case
excludes the operator; the dtype is never changed to avoid an exclusion. Nothing is re-run here.

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
from tilebench.llm.v2.metrics.baselines import label_matches  # noqa: E402
from tilebench.llm.v2.tasks import case_sets as cs_mod  # noqa: E402
from tilebench.llm.v2.tasks.representative import selected  # noqa: E402

OUT = REPO / "tilebench" / "llm" / "v2" / "manifests" / "trn2_eligibility.json"
CSV_ROOT = REPO / "results" / "TRN2" / "csv"
SCHEMA = "tilebench-llm-v2-trn2-eligibility/1"
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


def build() -> dict:
    sets = cs_mod.load_manifest()["operators"]
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
        eligible = n == 20 and n_valid == n
        bad = [c for c in cases if not c["baseline_valid"]]
        reasons = sorted({c.get("na_reason") or c.get("problem") or c.get("torch_status") or "invalid" for c in bad})
        if eligible:
            reason = f"20/20 configured {dtype} cases have a valid formal PyTorch eager baseline on the native Trn2 stack"
        else:
            reason = (f"{EXPLANATION.get(op, 'baseline not valid on every case')}: {n_valid}/{n} configured {dtype} "
                      f"cases have a valid baseline; CSV reasons {reasons}")
        counts["eligible" if eligible else "excluded"] += 1
        ops[op] = {"dtype": dtype, "case_set_id": entry["case_set_id"], "n_cases": n,
                   "cases_matched": sum(1 for c in cases if c["matching_rows"] == 1), "baseline_valid_cases": n_valid,
                   "status": "eligible" if eligible else "excluded", "reason": reason,
                   "csv": {"path": str(path.relative_to(REPO)), "sha256": _sha256(path) if path.exists() else None,
                           "rows_for_dtype": len(rows)},
                   "cases": cases}
    body = {
        "schema": SCHEMA, "device": "Trn2", "dsl": "nki", "status": "frozen", "frozen_on": "2026-10-10",
        "approved_by": "study owner, Trn2 NKI Base preparation instruction (2026-10-10) §3: an operator whose selected "
                       "task has any CPU-fallback or otherwise invalid formal baseline case is excluded; 20/20 valid "
                       "baseline cases are required; the dtype is never changed",
        "rule": "eligible iff all 20 frozen cases of the representative dtype have exactly one row in "
                "results/TRN2/csv/<op>_default.csv with torch_status == 'ok' and finite positive torch_ms",
        "protocol": {"revision": 4, "operators": len(ops), "cases_per_task": 20,
                     "representative_dtypes_sha256": _sha256(REPO / "tilebench/llm/v2/manifests/representative_dtypes.yaml"),
                     "case_sets_sha256": _sha256(REPO / "tilebench/llm/v2/manifests/case_sets.yaml")},
        "source": {**SOURCE_COMMIT, "csv_root": "results/TRN2/csv",
                   "stack": "native PyTorch eager on torch.device('neuron') vs NKI 0.7, device busy sum, warmup 1 / repeat 3"},
        "counts": counts,
        "excluded": sorted(op for op, v in ops.items() if v["status"] == "excluded"),
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
