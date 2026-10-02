"""Formal completeness audit of results/GH200 for one mode (default|autotune). Read-only."""
import csv, json, math, sys, collections
from pathlib import Path
MODE = sys.argv[1]
R = Path("/projects/kzhou6/bcui2/research/tilebench/Tilebench-gh200/results/GH200")
C = Path(f"/tmp/tilebench-gh200-campaign/{MODE}")
EXPECT_SHA = "c882fe5074a82618f44ce456aa390b25102cb509"
exp = json.load(open("/tmp/tilebench-gh200-campaign/expected.json"))
rc = {l.split("\t")[0]: l.split("\t")[1] for l in open(C / "exit_codes.tsv")} if (C / "exit_codes.tsv").exists() else {}
BK = ["triton", "cutile", "tilelang"]
issues = collections.defaultdict(list); tot_rows = tot_exp = 0
ok_count = collections.Counter(); complete_ops = 0
def bad_num(x):
    try: v = float(x)
    except (TypeError, ValueError): return True
    return math.isnan(v) or math.isinf(v) or v <= 0
for op, e in exp.items():
    tag = f"{op}_{MODE}_triton-cutile-tilelang"
    jp, cp, pp = R / "logs/time_measurement_logs" / f"{tag}.json", R / "csv" / f"{op}_{MODE}.csv", R / "logs/provenance" / f"{tag}.json"
    problems = []
    if rc.get(op) != "0": problems.append(f"exit={rc.get(op)}")
    if not jp.exists() or not cp.exists() or not pp.exists():
        problems.append("missing " + ",".join(n for n, p in (("json", jp), ("csv", cp), ("provenance", pp)) if not p.exists()))
        issues[op] += problems; tot_exp += e["n"]; continue
    rows = json.load(open(jp)); crow = list(csv.DictReader(open(cp)))
    tot_rows += len(rows); tot_exp += e["n"]
    if len(rows) != e["n"]: problems.append(f"json rows {len(rows)} != expected {e['n']}")
    if len(crow) != e["n"]: problems.append(f"csv rows {len(crow)} != expected {e['n']}")
    got = collections.Counter(r["dtype"] for r in rows)
    if dict(got) != e["per_dtype"]: problems.append(f"dtype coverage {dict(got)} != {e['per_dtype']}")
    keys = collections.Counter((json.dumps(r["params"], sort_keys=True), r["dtype"]) for r in rows)
    dups = [k for k, v in keys.items() if v > 1]
    if dups: problems.append(f"{len(dups)} duplicate (params,dtype)")
    ckeys = collections.Counter((r["params"], r["dtype"]) for r in crow)
    if any(v > 1 for v in ckeys.values()): problems.append("duplicate csv rows")
    for r in rows:
        where = f"{r['dtype']} {r['params']}"
        if bad_num(r.get("torch_ms")): problems.append(f"torch_ms bad {where}: {r.get('torch_ms')}")
        for b in BK:
            if r[b + "_ok"]: ok_count[b] += 1
            else: problems.append(f"{b} FAIL {where}: {str(r[b + '_err'])[:140]}")
            if r[b + "_ok"] and bad_num(r.get(b + "_ms")): problems.append(f"{b}_ms bad {where}: {r.get(b + '_ms')}")
    for r in crow:
        for col in ["torch_ms"] + [f"{b}_ms" for b in BK]:
            if bad_num(r.get(col)): problems.append(f"csv {col} bad {r['dtype']} {r['params']}: {r.get(col)}")
    prov = json.load(open(pp)); src = prov.get("source", prov)
    if src.get("git_sha") != EXPECT_SHA or src.get("dirty") or src.get("tracked_dirty"):
        problems.append(f"provenance sha={src.get('git_sha')} dirty={src.get('dirty')} tracked_dirty={src.get('tracked_dirty')}")
    if MODE == "autotune":
        at = json.load(open(R / "logs/autotune_logs" / f"{tag}.json"))
        for r in at:
            for b in BK:
                if not r.get(f"{b}_autotune_cfg"): problems.append(f"{b} no winner {r['dtype']} {r['params']}")
    if problems: issues[op] += problems
    else: complete_ops += 1
print(f"MODE={MODE}  operators complete: {complete_ops}/{len(exp)}  rows: {tot_rows}/{tot_exp}")
for b in BK: print(f"  {b:9s} PASS {ok_count[b]}  FAIL {tot_rows - ok_count[b]}")
for op, ps in issues.items():
    print(f"  ISSUE {op}: {len(ps)}"); [print("     ", p) for p in ps[:8]]
