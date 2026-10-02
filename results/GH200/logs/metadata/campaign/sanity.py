import json, sys, math, collections
from pathlib import Path
MODE = sys.argv[1]; R = Path("/projects/kzhou6/bcui2/research/tilebench/Tilebench-gh200/results/GH200/logs/time_measurement_logs")
flags = []; nseries = 0; disp = []
for jp in sorted(R.glob(f"*_{MODE}_triton-cutile-tilelang.json")):
    op = jp.name.split(f"_{MODE}_")[0]
    rows = json.load(open(jp))
    series = collections.defaultdict(list)
    for r in rows:
        for b in ("torch", "triton", "cutile", "tilelang"):
            series[(r["dtype"], b)].append((r.get("problem_size") or 0, r[f"{b}_ms"], r))
            st = r.get(f"{b}_stats") or {}
            if st.get("min") and st.get("max") and st["min"] > 0 and st["max"] / st["min"] > 3:
                disp.append((op, r["dtype"], b, r["params"], round(st["max"] / st["min"], 1)))
        for b in ("triton", "cutile", "tilelang"):
            sp = r.get(f"speedup_{b}")
            if sp and (sp > 200 or sp < 0.005): flags.append((op, r["dtype"], b, r["params"], f"extreme speedup {sp:.3g}"))
    for (dt, b), pts in series.items():
        nseries += 1; pts.sort(key=lambda p: p[0])
        for i in range(1, len(pts) - 1):
            l, m, h = pts[i - 1][1], pts[i][1], pts[i + 1][1]
            if min(l, m, h) > 0 and ((m > 10 * l and m > 10 * h) or (m * 10 < l and m * 10 < h)):
                flags.append((op, dt, b, pts[i][2]["params"], f"neighbor outlier {l:.4g} | {m:.4g} | {h:.4g} ms"))
print(f"{MODE}: series checked {nseries}; neighbor/speedup flags {len(flags)}; high max/min dispersion (>3x) {len(disp)}")
for f in flags[:40]: print("  FLAG", f)
for d in disp[:15]: print("  DISP", d)
