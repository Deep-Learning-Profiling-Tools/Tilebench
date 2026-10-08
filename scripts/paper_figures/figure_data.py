"""Shared data access + aggregation for the figure scripts. Reads ONLY artifacts/paper_figures/combined/.

Protocol (comparison_manifest.json):
  S[o,b,d]  = GM over valid autotune cases of torch_ms / dsl_ms
  S[b,d]    = GM over operators of S[o,b,d]          (category: GM over the category's operators)
  R[X/T]    = GM over cases valid for both of X_ms / triton_ms
  delta     = log2(S_dev2 / S_dev1) with both S computed over the SAME matched case_id_v2 set
"""
import csv
import gzip
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMBINED = REPO / "artifacts" / "paper_figures" / "combined"
PLOTS = REPO / "artifacts" / "paper_figures" / "plots"
DSL_SUPPORT = {"B200": ("triton", "cutile", "tilelang"), "GH200": ("triton", "cutile", "tilelang"), "MI300X": ("triton",)}

csv.field_size_limit(1 << 30)


def read_csv(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", newline="") as f:
        return list(csv.DictReader(f))


def gm(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    assert all(x > 0 and math.isfinite(x) for x in xs), "non-positive or non-finite value in geometric mean"
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_head():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def input_hashes(names):
    return {n: sha256(COMBINED / n) for n in names}


class Data:
    def __init__(self, mode="autotune"):
        self.mode = mode
        self.rows = [r for r in read_csv(COMBINED / "benchmark_cases_normalized.csv.gz") if r["mode"] == mode]
        self.categories = {r["operator"]: r["category"] for r in read_csv(COMBINED / "category_mapping.csv")}
        self.cat_order = sorted({(int(r["category_order"]), r["category"]) for r in read_csv(COMBINED / "category_mapping.csv")})
        self.cat_order = [c for _, c in self.cat_order]
        self.manifest = json.load(open(COMBINED / "comparison_manifest.json"))
        # (device, dsl, op) -> {case_id_v2: (torch_ms, dsl_ms, dtype, params_full_json)}
        self.cases = defaultdict(dict)
        self.excluded = []
        for r in self.rows:
            if r["validity"] != "valid":
                self.excluded.append({k: r[k] for k in ("device", "dsl", "operator", "dtype", "case_id_v2", "validity", "validity_source")})
                continue
            self.cases[(r["device"], r["dsl"], r["operator"])][r["case_id_v2"]] = (
                float(r["torch_ms"]), float(r["dsl_ms"]), r["dtype"], r["params_full_json"])

    def operators(self):
        return sorted(self.categories, key=lambda o: (self.cat_order.index(self.categories[o]), o))

    def supported(self, dev, dsl):
        return dsl in DSL_SUPPORT[dev]

    def speedup_op(self, dev, dsl, op, case_ids=None):
        c = self.cases.get((dev, dsl, op), {})
        ids = c.keys() if case_ids is None else [i for i in case_ids if i in c]
        vals = [c[i][0] / c[i][1] for i in ids]
        return gm(vals), len(vals)

    def speedup_group(self, dev, dsl, ops):
        per = [self.speedup_op(dev, dsl, o)[0] for o in ops]
        return gm(per), sum(1 for p in per if p is not None)

    def ratio_op(self, dev, num, den, op):
        a, b = self.cases.get((dev, num, op), {}), self.cases.get((dev, den, op), {})
        ids = sorted(set(a) & set(b))
        return gm([a[i][1] / b[i][1] for i in ids]), len(ids)

    def latency_gm(self, dev, dsl, op, case_ids):
        c = self.cases.get((dev, dsl, op), {})
        return gm([c[i][1] for i in case_ids if i in c])

    def matched_ids(self, op, pairs):
        """case_id_v2 valid for every (device, dsl) in pairs."""
        sets = [set(self.cases.get((d, s, op), {})) for d, s in pairs]
        return sorted(set.intersection(*sets)) if sets else []

    def delta_op(self, dsl, dev1, dev2, op):
        ids = self.matched_ids(op, [(dev1, dsl), (dev2, dsl)])
        if not ids:
            return None, 0
        s1, _ = self.speedup_op(dev1, dsl, op, ids)
        s2, _ = self.speedup_op(dev2, dsl, op, ids)
        return math.log2(s2 / s1), len(ids)

    def case(self, dev, dsl, op, case_id):
        return self.cases.get((dev, dsl, op), {}).get(case_id)
