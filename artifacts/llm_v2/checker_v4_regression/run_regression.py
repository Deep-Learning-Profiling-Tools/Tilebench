"""Offline checker-v4 regression on every archived candidate of the excluded
campaigns (read-only: no API, no GPU, nothing written into the campaigns).
Per candidate: the verdict without and with the task's frozen scalar input
positions (tasks.input_kinds), i.e. the only change from checker v3 to v4.
Lists every candidate whose verdict changes and every review_required /
confirmed_violation under v4.

    python artifacts/llm_v2/checker_v4_regression/run_regression.py --out <json> \\
        --campaign <name>=<campaign output dir> [--campaign ...]
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
    from tilebench.llm.v2.tasks.input_kinds import scalar_positions
    from tilebench.llm.v2.validation.contract_checks import check_compliance
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--campaign", action="append", required=True, help="<campaign name>=<campaign output dir>")
    a = ap.parse_args()
    rules, rows = {}, []
    for name, root in (x.split("=", 1) for x in a.campaign):
        for tf in sorted(Path(root).glob("base/B200/*/*/*/*/*/trajectory.json")):
            st = json.loads(tf.read_text()); t = st["task"]
            r_ = rules.setdefault(t["operator"], load_contract(t["operator"], require_approved=False).rules)
            pos = scalar_positions(t["operator"])
            for r in st["rounds"]:
                for at in r["attempts"]:
                    src = tf.parent / f"round_{r['round']:02d}" / f"attempt_{at['attempt']}" / st["output_file"]
                    if not src.exists():
                        continue
                    text = src.read_text()
                    before = check_compliance(text, t["dsl"], r_)
                    after = check_compliance(text, t["dsl"], r_, scalar_positions=pos)
                    rows.append({"campaign": name, "dsl": t["dsl"], "model": st["model"], "op": t["operator"], "round": r["round"],
                                 "candidate_sha256": hashlib.sha256(text.encode()).hexdigest(), "scalar_positions": sorted(pos),
                                 "v3": before.verdict, "v4": after.verdict,
                                 "v3_review_items": before.review_items(), "v4_review_items": after.review_items(),
                                 "v4_confirmed": after.diagnostics()})
    trans = collections.Counter(f"{x['campaign']}: {x['v3']} -> {x['v4']}" for x in rows)
    rank = {"clear": 0, "audit_only": 1, "review_required": 2, "confirmed_violation": 3}
    stricter = [x for x in rows if rank.get(x["v4"], 9) > rank.get(x["v3"], 9)]
    out = {"checker": checker_fingerprint(), "candidates": len(rows), "transitions": dict(sorted(trans.items())),
           "stricter_under_v4": len(stricter),
           "changed": [{k: x[k] for k in ("campaign", "dsl", "model", "op", "round", "v3", "v4", "v3_review_items", "v4_review_items")}
                       for x in rows if x["v3"] != x["v4"]],
           "v4_review_or_violation": [{k: x[k] for k in ("campaign", "dsl", "model", "op", "round", "v4", "v4_review_items", "v4_confirmed")}
                                      for x in rows if x["v4"] in ("review_required", "confirmed_violation")]}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("candidates", "transitions", "stricter_under_v4")}, indent=1))
    print("v4 review/violation:", [(x["campaign"][:30], x["dsl"], x["model"], x["op"], x["round"], x["v4"]) for x in out["v4_review_or_violation"]])
    return 0 if not stricter else 1


if __name__ == "__main__":
    sys.exit(main())
