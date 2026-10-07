"""Offline replay of the 16 archived candidates that the revision-3 evaluator
closed as `contract_violation` with `cuda.matmul.fp32_precision: ('none', 'ieee')`
(pilot campaign formal_b200_base_1dtype5r_2026-10-06, read-only).

Each candidate is re-evaluated with the FIXED evaluator (PrecisionGuard on the
canonical fp32_precision family) on the single case it was generated for (the
revision-3 task case), in the bwrap sandbox under the device lock; nothing is
written into the pilot campaign. Acceptance: none of them is a violation because
of the evaluator-induced `none -> ieee` change (other outcomes are recorded as
they are).

    env -u LD_LIBRARY_PATH python artifacts/llm_v2/precision_guard_regression/run_replay.py --out <json> \
        --pilot <formal_b200_base_1dtype5r_2026-10-06 campaign dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main() -> int:
    from tilebench import hardware
    from tilebench.llm.v2.contracts.loader import load_contract
    from tilebench.llm.v2.evaluation.job import build_evaluation_job
    from tilebench.llm.v2.evaluation.launcher import SubprocessEvaluator
    from tilebench.llm.v2.manifests import schema as ms
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", required=True, help="output dir of formal_b200_base_1dtype5r_2026-10-06")
    ap.add_argument("--sandbox-root", default=str(REPO / "outputs/llm_v2/_precision_replay"))
    a = ap.parse_args()
    pilot = Path(a.pilot)
    study = ms.load_study()
    ev = SubprocessEvaluator(device="B200", timeout_s=int(study["evaluation"]["worker_timeout_s"]),
                             sandbox_root=Path(a.sandbox_root), isolation="bwrap", lock_root=REPO / "outputs/llm_v2")
    rows = []
    for tf in sorted(pilot.glob("base/B200/*/*/*/*/*/trajectory.json")):
        st = json.loads(tf.read_text())
        for r in st["rounds"]:
            diag = str(r.get("diagnostic") or "")
            if r.get("status") != "contract_violation" or "fp32_precision" not in diag:
                continue
            a_ = r["attempts"][-1]
            src = Path(a_["source_path"])
            t = st["task"]
            job = build_evaluation_job(operator=t["operator"], dtype=t["dtype"], params=t["params"], dsl=t["dsl"], device="B200",
                                       arch=hardware.detect_arch(), rules=load_contract(t["operator"]).rules, study=study,
                                       identity={"replay": "precision_guard_regression"})
            t0 = time.time()
            res = ev.evaluate(src, job, r["round"], 1)
            c = (res.get("cases") or [{}])[0]
            rows.append({"trajectory": str(tf.parent.relative_to(pilot)), "dsl": t["dsl"], "model": st["model"], "operator": t["operator"],
                         "round": r["round"], "candidate_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                         "recorded_status": r["status"], "recorded_diagnostic": diag[:300],
                         "replay_status": res.get("status"), "replay_case_status": c.get("status"),
                         "replay_diagnostic": str(res.get("diagnostic") or "")[:600],
                         "precision_violation": "precision/backend state changed" in str(res.get("diagnostic") or ""),
                         "precision_guard_family": (res.get("precision_guard") or {}).get("family"),
                         "precision_guard_restores": (res.get("precision_guard") or {}).get("restore_count"),
                         "elapsed_s": round(time.time() - t0, 1)})
            print(json.dumps({k: rows[-1][k] for k in ("dsl", "model", "operator", "round", "replay_status", "precision_violation")}),
                  flush=True)
    out = {"schema": "tilebench-llm-v2-precision-guard-regression/1", "pilot": pilot.name, "candidates": len(rows),
           "precision_violations_after_fix": sum(1 for r in rows if r["precision_violation"]),
           "replay_status_counts": {s: sum(1 for r in rows if r["replay_status"] == s) for s in sorted({r["replay_status"] for r in rows})},
           "accepted": len(rows) == 16 and not any(r["precision_violation"] for r in rows), "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("candidates", "precision_violations_after_fix", "replay_status_counts", "accepted")}))
    return 0 if out["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
