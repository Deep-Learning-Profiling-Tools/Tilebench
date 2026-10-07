"""Offline checker-v3 regression on every archived candidate of the two excluded
pilots (read-only: no API, no GPU, nothing written into the pilots). Records, per
candidate, the verdict recorded by the checker in force at the time and the
verdict of the current checker; lists every review_required / confirmed_violation
of either.

    python artifacts/llm_v2/checker_v3_regression/run_regression.py --out <json> \
        --pilot formal_b200_base_2026-10-06=<campaign dir> --pilot formal_b200_base_1dtype5r_2026-10-06=<campaign dir>
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main():
    from tilebench.llm.v2.contracts.loader import load_contract
    from tilebench.llm.v2.evaluation.fingerprint import checker_fingerprint
    from tilebench.llm.v2.validation.contract_checks import check_compliance
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", action="append", required=True, help="<campaign name>=<campaign output dir>"); a = ap.parse_args()
    rules, rows = {}, []
    for name, root in (x.split("=", 1) for x in a.pilot):
        root = Path(root)
        for tf in sorted(root.glob("base/B200/*/*/*/*/*/trajectory.json")):
            st = json.loads(tf.read_text()); t = st["task"]
            r_ = rules.setdefault(t["operator"], load_contract(t["operator"], require_approved=False).rules)
            for r in st["rounds"]:
                for at in r["attempts"]:
                    src = tf.parent / f"round_{r['round']:02d}" / f"attempt_{at['attempt']}" / st["output_file"]
                    if not src.exists():
                        continue
                    text = src.read_text()
                    res = check_compliance(text, t["dsl"], r_)
                    rows.append({"campaign": name, "dsl": t["dsl"], "model": st["model"], "op": t["operator"], "round": r["round"],
                                 "candidate_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                 "old": (at.get("compliance") or {}).get("verdict"), "new": res.verdict,
                                 "review_items": res.review_items(), "confirmed": res.diagnostics()})
    trans = collections.Counter(f"{x['campaign']}: {x['old']} -> {x['new']}" for x in rows)
    out = {"checker": checker_fingerprint(), "candidates": len(rows), "transitions": dict(sorted(trans.items())),
           "new_review_or_confirmed": [x for x in rows if x["new"] in ("review_required", "confirmed_violation")],
           "old_review_or_confirmed": [{k: x[k] for k in ("campaign", "dsl", "model", "op", "round", "old", "new")}
                                       for x in rows if x["old"] in ("review_required", "confirmed_violation")]}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("candidates", "transitions")}, indent=1))
    print("new review/confirmed:", [(x["campaign"][-12:], x["dsl"], x["model"], x["op"], x["round"], x["new"], (x["review_items"] or x["confirmed"])[:1])
                                    for x in out["new_review_or_confirmed"]])


if __name__ == "__main__":
    main()
