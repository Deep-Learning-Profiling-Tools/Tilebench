"""Offline checker-v2 regression on the archived candidates of the revision-2 pilot
formal_b200_base_2026-10-06 (read-only: no API, no GPU, nothing written into the pilot)."""
import sys, json, collections, hashlib
sys.dont_write_bytecode = True
from pathlib import Path
WT = Path("/projects/kzhou6/bcui2/research/tilebench/llm_wt")
sys.path.insert(0, str(WT))
from tilebench.llm.v2.validation.contract_checks import check_compliance, CHECKER_VERSION
from tilebench.llm.v2.contracts.loader import load_contract
CAMP = Path("/projects/kzhou6/bcui2/research/tilebench/llm_b200_wt/outputs/llm_v2/formal_b200_base_2026-10-06")
rows = []
rules_cache = {}
for tf in sorted(CAMP.glob("base/B200/*/*/*/*/*/trajectory.json")):
    d = json.loads(tf.read_text()); tdir = tf.parent
    dsl, model, op, dt = tf.relative_to(CAMP).parts[2:6]
    if op not in rules_cache:
        rules_cache[op] = load_contract(op, require_approved=False).rules
    for r in d["rounds"]:
        for a in r["attempts"]:
            src = tdir / f"round_{r['round']:02d}" / f"attempt_{a['attempt']}" / d["output_file"]
            if not src.exists():
                continue
            text = src.read_text()
            old = (a.get("compliance") or {}).get("verdict")
            res = check_compliance(text, dsl, rules_cache[op])
            rows.append({"traj": d["trajectory_id"], "dsl": dsl, "model": model, "op": op, "dtype": dt, "round": r["round"],
                         "attempt": a["attempt"], "candidate_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "old": old, "new": res.verdict, "review_items": res.review_items(), "confirmed": res.diagnostics(),
                         "audit_flags": res.audit_flags(), "traj_status": d["status"]})
out = {"checker_version": CHECKER_VERSION, "n": len(rows),
       "transitions": collections.Counter(f"{x['old']} -> {x['new']}" for x in rows),
       "old16": [x for x in rows if x["traj_status"] == "review_required" and x["old"] == "review_required"
                 and x["round"] == max(rr["round"] for rr in json.loads((CAMP/'base/B200'/x['dsl']/x['model']/x['op']/x['dtype']).glob('*/trajectory.json').__next__().read_text())["rounds"])]}
print(json.dumps({"n": out["n"], "transitions": out["transitions"]}, indent=1))
rev = [x for x in rows if x["new"] == "review_required"]
conf_new = [x for x in rows if x["new"] == "confirmed_violation" and x["old"] != "confirmed_violation"]
print("REVIEW reasons:", collections.Counter(i.split(": ",1)[-1][:110] for x in rev for i in x["review_items"]).most_common(20))
print("new CONFIRMED (old not confirmed):", [(x["dsl"],x["op"],x["round"],x["old"],x["confirmed"][:2]) for x in conf_new][:20])
print("old16:", [(x["dsl"], x["model"], x["op"], x["dtype"], x["round"], x["new"]) for x in out["old16"]])
print("AUDIT flag kinds:", collections.Counter(i.split(": ",1)[-1].split(" [")[0][:80] for x in rows for i in x["audit_flags"]).most_common(25))
Path(sys.argv[1]).write_text(json.dumps({"checker_version": CHECKER_VERSION, "rows": rows,
                                         "transitions": out["transitions"]}, indent=1, default=str))
