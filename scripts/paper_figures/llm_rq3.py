"""LLM Base campaigns of the four devices (paper RQ3): per-round table, 16-row summary, BestSOL(B) figure, appendix
tables and the human-kernel comparison.

Paper metric (largest frozen input case k of each operator's representative dtype). It is computed here, independently
of tilebench.llm.v2.metrics.efficiency, which scores the 20-case geometric mean:
  valid_i     round i passed all 20 frozen cases under the adopted evaluation revision of the round
  SOL_i       T_SOL,k / T_i,k with T_SOL,k from the frozen empirical scoring table of the device (values > 1 kept)
  C_i         cumulative input + output tokens of rounds 1..i of the operator, failed rounds included
  BestSOL(B)  max({SOL_i : C_i <= B, valid_i} U {0}); averaged with equal weight over the eligible operators
              (45 per device and DSL, 36 for Trn2 NKI); an operator without a valid round counts 0

Reads only: the extracted campaign archives (--raw), the frozen manifests and scoring tables via `git show` at the
pinned device-branch commits, and the human-kernel result CSVs. Writes artifacts/paper_figures/llm_rq3/.

  python scripts/paper_figures/llm_rq3.py --raw <dir holding B200/ GH200/ MI300X/ Trainium2/ with x/ extracted>
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style as PS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "paper_figures" / "llm_rq3"
M = "tilebench/llm/v2/manifests"
# device -> (branch commit, HF archive folder, campaign, DSLs, frozen scoring table)
DEVICES = {
    "B200": ("ac183b885a344dd193488d5850aeb6d4f0e41e26", "B200", "formal_b200_base_rev4_reffix_2026-10-07",
             ("triton", "cutile", "tilelang"), "artifacts/llm_v2/scoring/B200/scoring_table_B200_rev4_casesets_2026-10-07.json"),
    "GH200": ("436d033fdcc70a06e9e4480b61e73da103ecd418", "GH200", "formal_gh200_base_rev4_reffix_2026-10-07",
              ("triton", "cutile", "tilelang"), "artifacts/llm_v2/scoring/GH200/scoring_table_GH200_rev4_casesets_2026-10-07.json"),
    "MI300X": ("be5f8111f2bd81be2dbcf6144f9d4f86849100d5", "MI300X", "formal_mi300x_triton_base_rev4_2026-10-09",
               ("triton",), "artifacts/llm_v2/scoring/MI300X/scoring_table_MI300X_rev4_casesets_2026-10-09.json"),
    "Trn2": ("7f2f0ca20575ac72cb105603cda811927d54c2e4", "Trainium2", "formal_trn2_nki_base_rev4_2026-10-10",
             ("nki",), "artifacts/llm_v2/scoring/Trn2/scoring_table_Trn2_rev4_casesets_2026-10-10.json"),
}
MAIN_COMMIT = "00a9751dfcd5bfa5ad15f12775c03e0b506ff213"       # origin/main holding results/TRN2/csv (#322)
TRACKS = [(d, s) for d, v in DEVICES.items() for s in v[3]]       # the 8 device/DSL combinations
MODELS = ("gpt", "claude")
MODEL_LABEL = {"gpt": "GPT (gpt-6.1-sol)", "claude": "Claude (claude-opus-5-5)"}
DSL_LABEL = {**PS.DSL_LABEL, "nki": "NKI"}
N_ROUNDS = 5
HUMAN_CSV = {"B200": "results/B200/csv/{op}_autotune.csv", "GH200": "results/GH200/csv/{op}_autotune.csv",
             "MI300X": "results/MI300X/csv/{op}_autotune.csv"}
HUMAN_COL = {"triton": "triton_ms", "cutile": "cutile_ms", "tilelang": "tilelang_ms"}


def git_show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO, check=True, capture_output=True).stdout


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


# ------------------------------------------------------------------------------------------------ frozen inputs
def load_frozen():
    commit = DEVICES["B200"][0]
    raw = {n: git_show(commit, f"{M}/{n}") for n in ("case_sets.yaml", "representative_dtypes.yaml", "api_pricing.yaml")}
    for dev, (c, *_rest) in DEVICES.items():                        # the frozen manifests are identical on every branch
        for n, b in raw.items():
            assert git_show(c, f"{M}/{n}") == b, (dev, n)
    case_sets = yaml.safe_load(raw["case_sets.yaml"])["operators"]
    rep = yaml.safe_load(raw["representative_dtypes.yaml"])["operators"]
    case_k = {}
    for op, cs in case_sets.items():
        best = None
        for c in cs["cases"]:                                         # largest problem size; ties keep the last case
            if best is None or c["problem_size"] >= best["problem_size"]:
                best = c
        case_k[op] = {"dtype": cs["dtype"], "case_set_id": cs["case_set_id"], "case_ids": [c["case_id"] for c in cs["cases"]],
                      "case_id": best["case_id"], "case_index": best["case_index"], "params": best["params"],
                      "problem_size": best["problem_size"],
                      "matches_representative_case": rep[op]["dtype"] == cs["dtype"] and rep[op]["case"]["case_id"] == best["case_id"]}
    trn2 = json.loads(git_show(DEVICES["Trn2"][0], f"{M}/trn2_eligibility.json"))
    eligible = {d: sorted(case_k) for d in DEVICES}
    eligible["Trn2"] = sorted(o for o, v in trn2["operators"].items() if v["status"] == "eligible")
    assert len(eligible["Trn2"]) == trn2["counts"]["eligible"] == 36
    scoring, tables = {}, {}
    for dev, (c, _a, _cmp, _dsls, path) in DEVICES.items():
        b = git_show(c, path)
        t = json.loads(b)
        tables[dev] = {"path": path, "commit": c, "sha256": sha256_bytes(b), "profile_sha256": t["profile_entry"]["sha256"],
                       "calibration_id": t["calibration_id"], "arithmetic_modes_sha256": t["arithmetic_modes_sha256"],
                       "status_counts": t["status_counts"]}
        for r in t["rows"]:
            assert r["status"] == "ok", (dev, r["operator"], r["case_id"])
            scoring[(dev, r["operator"], r["case_id"])] = r["t_emp_ms"]
    frozen = {"commits": {d: v[0] for d, v in DEVICES.items()},
              "manifest_sha256": {n: sha256_bytes(b) for n, b in raw.items()}, "scoring_tables": tables}
    return case_k, eligible, scoring, frozen


# ------------------------------------------------------------------------------------------------ raw trajectories
def load_rounds(raw_root, case_k, scoring, frozen):
    rows, archives = [], {}
    for dev, (_c, folder, campaign, dsls, _p) in DEVICES.items():
        sha_file = Path(raw_root) / folder / f"{campaign}.tar.zst.sha256"
        archives[dev] = {"hf_path": f"{folder}/{campaign}.tar.zst", "sha256": sha_file.read_text().split()[0]}
        base = Path(raw_root) / folder / "x" / "outputs" / "llm_v2" / campaign / "base"
        for tp in sorted(base.glob("*/*/*/*/*/*/trajectory.json")):
            t = json.loads(tp.read_text())
            assert t["run_type"] == "formal" and t["condition"] == "base" and t["status"] == "complete", tp
            sb = t["scoring_binding"]
            assert sb["profile_sha256"] == frozen["scoring_tables"][dev]["profile_sha256"], (dev, tp)
            assert sb["arithmetic_modes_sha256"] == frozen["scoring_tables"][dev]["arithmetic_modes_sha256"], (dev, tp)
            assert t["pricing_binding"]["pricing_snapshot_sha256"] == frozen["manifest_sha256"]["api_pricing.yaml"], tp
            task = t["task"]
            op, dsl, model = task["operator"], task["dsl"], t["model"]
            k = case_k[op]
            assert dsl in dsls and task["dtype"] == k["dtype"] and task["case_set_id"] == k["case_set_id"], tp
            assert task["case_ids"] == k["case_ids"] and len(t["rounds"]) == N_ROUNDS, tp
            t_sol = scoring[(dev, op, k["case_id"])]
            cum_tok, tok_lb, cum_usd_lb, usd_unknown = 0, False, 0.0, False
            for r in t["rounds"]:
                assert len(r["attempts"]) == 1, tp
                a = r["attempts"][0]
                u = a["usage"] or {}
                tin, tout = u.get("logical_input_total"), u.get("logical_output_total")
                tot = u.get("logical_total")
                tr = a["transport"]
                n_unknown = sum(1 for x in tr if x.get("charged") == "unknown")
                tok_lb = tok_lb or n_unknown > 0 or tot is None
                cum_tok += tot or 0
                usd_unknown = usd_unknown or a["usd_cost_status"] != "exact_from_usage_and_frozen_price"
                cum_usd_lb += a["usd_cost_lower_bound"] or 0.0
                rdir = tp.parent / f"round_{r['round']:02d}" / f"attempt_{a['attempt']}"
                rev = r["evaluation_revision"]
                row = {"device": dev, "dsl": dsl, "model": model, "model_id": a["model_id"], "operator": op, "dtype": k["dtype"],
                       "round": r["round"], "trajectory_id": t["trajectory_id"], "status": r["status"],
                       "valid": int(r["status"] == "valid" and r["valid_cases"] == 20 and r["cases_total"] == 20),
                       "valid_cases": r["valid_cases"], "case_k_id": k["case_id"], "case_k_index": k["case_index"],
                       "case_k_problem_size": k["problem_size"], "case_k_params": json.dumps(k["params"], sort_keys=True),
                       "t_sol_k_ms": t_sol, "latency_k_ms": None, "case_k_status": None, "sol_k": None,
                       "latency_geomean20_ms": r["latency_ms_geomean"], "evaluation_revision": rev,
                       "evaluation_revisions": ";".join(r["evaluation_revisions"] or []), "other_revisions_latency_k_ms": "",
                       "input_tokens": tin, "output_tokens": tout, "reasoning_tokens": u.get("reasoning_or_thinking_output") or (0 if u.get("reasoning_output") == 0 else None),
                       "tokens": tot, "cum_tokens": cum_tok, "cum_tokens_is_lower_bound": int(tok_lb),
                       "usd": a["usd_cost"], "usd_lower_bound": a["usd_cost_lower_bound"], "usd_status": a["usd_cost_status"],
                       "cum_usd": None if usd_unknown else round(cum_usd_lb, 6), "cum_usd_lower_bound": round(cum_usd_lb, 6),
                       "transport_attempts": len(tr), "transport_failed_unknown_charge": n_unknown,
                       "api_elapsed_s": sum(x["elapsed_s"] for x in tr),
                       "api_elapsed_success_s": sum(x["elapsed_s"] for x in tr if x.get("event") == "succeeded"),
                       "terminal_status": a["terminal_status"], "truncated": int(bool(a["truncated"])), "compliance_verdict": a["verdict"],
                       "reviews": len(r["reviews"] or []), "eval_elapsed_s": None, "eval_worker_wall_s": None,
                       "eval_device_lock_wait_s": None, "timing_execution_mode": r["timing_execution_mode"],
                       "request_end_t": max((x["t"] for x in tr if "t" in x), default=None),
                       "request_start_t": min((x["t"] - x["elapsed_s"] for x in tr if "t" in x), default=None), "eval_end_t": None}
                if rev:
                    ev = json.loads((rdir / rev / "evaluation.json").read_text())
                    meta = json.loads((rdir / rev / "META.json").read_text())
                    assert ev["status"] == r["status"] or r["status"] == "contract_violation", (tp, r["round"], ev["status"], r["status"])
                    assert ev["valid_cases"] == r["valid_cases"] and ev.get("evaluation_revision", rev) == rev, (tp, r["round"])
                    ck = next(c for c in ev["cases"] if c["case_id"] == k["case_id"]) if any(c["case_id"] == k["case_id"] for c in ev["cases"]) else None
                    if ck is not None:
                        row["case_k_status"] = ck["status"]
                        row["latency_k_ms"] = ck.get("latency_ms_mean")
                    row["eval_elapsed_s"], row["eval_worker_wall_s"] = ev.get("elapsed_s"), ev.get("worker_wall_s")
                    row["eval_device_lock_wait_s"] = ev.get("device_lock_wait_s")
                    row["eval_end_t"] = meta["created"] + (ev.get("device_lock_wait_s") or 0) + (ev.get("worker_wall_s") or 0)
                    old = []                  # every other revision on disk, kept for audit; '(unregistered)' = not in the trajectory
                    for od in sorted(rdir.glob("eval_*")):
                        o = od.name
                        if o != rev:
                            oe = json.loads((od / "evaluation.json").read_text())
                            oc = [c for c in oe["cases"] if c["case_id"] == k["case_id"]]
                            tag = "" if o in (r["evaluation_revisions"] or []) else "(unregistered)"
                            old.append(f"{o}{tag}:{oc[0].get('latency_ms_mean') or '' if oc else ''}")
                    row["other_revisions_latency_k_ms"] = ";".join(old)
                if row["valid"]:
                    assert row["case_k_status"] == "valid" and row["latency_k_ms"], (tp, r["round"])
                    row["sol_k"] = t_sol / row["latency_k_ms"]
                rows.append(row)
    return rows, archives


# ------------------------------------------------------------------------------------------------ metrics
def per_operator(rows, eligible):
    """(device, dsl, model, op) -> list of rounds sorted by round."""
    g = defaultdict(list)
    for r in rows:
        g[(r["device"], r["dsl"], r["model"], r["operator"])].append(r)
    for key, v in g.items():
        v.sort(key=lambda r: r["round"])
        assert [r["round"] for r in v] == list(range(1, N_ROUNDS + 1)), key
    for dev, dsl in TRACKS:
        for m in MODELS:
            ops = {k[3] for k in g if k[:3] == (dev, dsl, m)}
            assert ops == set(eligible[dev]), (dev, dsl, m, len(ops))
    return g


def best_sol(rounds, budget=math.inf):
    return max([r["sol_k"] for r in rounds if r["valid"] and r["cum_tokens"] <= budget] + [0.0])


def curve(g, dev, dsl, model, ops):
    """Exact step function of the mean BestSOL(B) over ops: breakpoints at every valid round's C_i."""
    pts = sorted({r["cum_tokens"] for o in ops for r in g[(dev, dsl, model, o)] if r["valid"]})
    xs, ys = [0], [0.0]
    for b in pts:
        xs.append(b)
        ys.append(sum(best_sol(g[(dev, dsl, model, o)], b) for o in ops) / len(ops))
    return xs, ys


def summary(g, eligible, ops_override=None):
    out = []
    for dev, dsl in TRACKS:
        for m in MODELS:
            ops = ops_override or eligible[dev]
            R = [r for o in ops for r in g[(dev, dsl, m, o)]]
            best = [best_sol(g[(dev, dsl, m, o)]) for o in ops]
            tok = [g[(dev, dsl, m, o)][-1]["cum_tokens"] for o in ops]
            mean_best, mean_tok = sum(best) / len(ops), sum(tok) / len(ops)
            usd_lb = sum(r["usd_lower_bound"] or 0 for r in R)
            n_unknown = sum(1 for r in R if r["usd_status"] != "exact_from_usage_and_frozen_price")
            out.append({"device": dev, "dsl": dsl, "model": m, "eligible_operators": len(ops), "rounds": len(R),
                        "valid_rounds": sum(r["valid"] for r in R), "valid_rate": sum(r["valid"] for r in R) / len(R),
                        "success_at_5": sum(1 for o in ops if any(r["valid"] for r in g[(dev, dsl, m, o)])) / len(ops),
                        "operators_with_valid_kernel": sum(1 for o in ops if any(r["valid"] for r in g[(dev, dsl, m, o)])),
                        "mean_bestsol_at_5": mean_best, "operators_sol_above_1": sum(1 for b in best if b > 1),
                        "mean_tokens_per_op": mean_tok, "tokens_lower_bound": int(any(r["cum_tokens_is_lower_bound"] for r in R)),
                        "sol_per_million_tokens": mean_best / (mean_tok / 1e6),
                        "estimated_usd": None if n_unknown else round(usd_lb, 2), "estimated_usd_lower_bound": round(usd_lb, 2),
                        "attempts_with_unknown_charge": n_unknown})
    return out


# ------------------------------------------------------------------------------------------------ human kernels
def parse_params(s):
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            out[k.strip()] = v.strip()
    return out


def human_latency(dev, dsl, op, k):
    """Formal human-kernel latency at the same operator, dtype and largest case k: (ms, source)."""
    def match(rows, col):
        hits = [r for r in rows if r.get("dtype") == k["dtype"]
                and all(k["params"].get(p) is not None and float(k["params"][p]) == v if isinstance(v, float) else str(k["params"].get(p)) == v
                        for p, v in parse_params(r["params"]).items())]
        vals = [float(r[col]) for r in hits if r.get(col) not in (None, "", "nan", "NaN")]
        return (vals[0], len(hits)) if len(vals) == 1 else (None, len(hits))
    if dev == "Trn2":
        try:
            rows = list(csv.DictReader(git_show(MAIN_COMMIT, f"results/TRN2/csv/{op}_default.csv").decode().splitlines()))
        except subprocess.CalledProcessError:
            return None, None, "no TRN2 CSV"
        v, n = match(rows, "nki_ms")
        sel = list(csv.DictReader(git_show(MAIN_COMMIT, "results/TRN2/csv/autotune_selected_cases/nki_autotune_largest_case_per_dtype.csv").decode().splitlines()))
        at, _ = match([r for r in sel if r["operator"] == op], "autotune_nki_ms")
        return v, at, f"results/TRN2/csv/{op}_default.csv nki_ms @ {MAIN_COMMIT[:8]} ({n} row)"
    p = HUMAN_CSV[dev].format(op=op)
    rows = list(csv.DictReader(git_show("HEAD", p).decode().splitlines()))
    v, n = match(rows, HUMAN_COL[dsl])
    return v, None, f"{p} {HUMAN_COL[dsl]} ({n} row)"


# ------------------------------------------------------------------------------------------------ figure
MODEL_STYLE = {"gpt": dict(color="#1F4E79", ls="-"), "claude": dict(color="#C8553D", ls="--")}


def plot_curves(g, eligible, out_root):
    PS.apply()
    W, H = PS.DOUBLE_COL_IN, 3.05
    fig, axes = plt.subplots(2, 4, figsize=(W, H), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.165, wspace=0.08, hspace=0.42)
    curves = {}
    ymax = 0
    for ax, (dev, dsl) in zip(axes.flat, TRACKS):
        for m in MODELS:
            xs, ys = curve(g, dev, dsl, m, eligible[dev])
            end = sum(g[(dev, dsl, m, o)][-1]["cum_tokens"] for o in eligible[dev]) / len(eligible[dev])
            xs_plot = [max(x, 1e3) for x in xs] + [2e6]
            ys_plot = ys + [ys[-1]]
            ax.step(xs_plot, ys_plot, where="post", lw=1.1, **MODEL_STYLE[m], label=MODEL_LABEL[m])
            curves[f"{dev}|{dsl}|{m}"] = {"budget_tokens": xs, "mean_bestsol": ys, "mean_tokens_per_op_at_5": end}
            ymax = max(ymax, max(ys))
        ax.set_xscale("log")
        ax.set_xlim(1e4, 2e6)
        ax.set_xticks([1e4, 1e5, 1e6])
        ax.set_xticklabels(["10k", "100k", "1M"])
        ax.minorticks_off()
        ax.grid(True, which="major", color=PS.GRID, lw=0.35)
        ax.set_title(f"{dev} {DSL_LABEL[dsl]}" + (f" ({len(eligible[dev])} ops)" if len(eligible[dev]) != 45 else ""),
                     fontsize=7.5, pad=2.5, loc="left")
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylim(0, math.ceil(ymax * 10 + 0.5) / 10)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean BestSOL(B)", fontsize=7.5)
    fig.text(0.535, 0.015, "token budget B per operator (cumulative input + output tokens, log scale)", ha="center", fontsize=7.5)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.535, 1.0))
    paths, layout = PS.save(fig, out_root, "figures", "fig_llm_bestsol_budget")
    plt.close(fig)
    return paths, layout, curves


# ------------------------------------------------------------------------------------------------ outputs
def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k])) for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory with <HF folder>/x/ (extracted archives) and the .sha256 files")
    a = ap.parse_args()
    case_k, eligible, scoring, frozen = load_frozen()
    rows, archives = load_rounds(a.raw, case_k, scoring, frozen)
    assert len(rows) == 3510, len(rows)
    g = per_operator(rows, eligible)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "llm_rounds.csv", rows)

    S = summary(g, eligible)
    write_csv(OUT / "llm_summary_16.csv", S)
    common = sorted(set.intersection(*(set(eligible[d]) for d in DEVICES)))
    write_csv(OUT / "appendix" / "llm_cross_device_common_ops.csv", summary(g, eligible, ops_override=common))

    # per-operator performance, with the human kernel at the same case k
    per_op, human = [], []
    for dev, dsl in TRACKS:
        for op in eligible[dev]:
            k = case_k[op]
            h_ms, h_at, h_src = human_latency(dev, dsl, op, k)
            t_sol = scoring[(dev, op, k["case_id"])]
            hrow = {"device": dev, "dsl": dsl, "operator": op, "dtype": k["dtype"], "case_k_id": k["case_id"],
                    "case_k_params": json.dumps(k["params"], sort_keys=True), "t_sol_k_ms": t_sol, "human_latency_k_ms": h_ms,
                    "human_sol_k": t_sol / h_ms if h_ms else None, "human_autotune_latency_k_ms": h_at, "human_source": h_src}
            for m in MODELS:
                R = g[(dev, dsl, m, op)]
                v = [r for r in R if r["valid"]]
                b = max(v, key=lambda r: r["sol_k"]) if v else None
                per_op.append({"device": dev, "dsl": dsl, "model": m, "operator": op, "dtype": k["dtype"], "case_k_id": k["case_id"],
                               "t_sol_k_ms": t_sol, "valid_rounds": len(v), "first_valid_round": v[0]["round"] if v else None,
                               "best_round": b["round"] if b else None, "best_latency_k_ms": b["latency_k_ms"] if b else None,
                               "bestsol_at_5": b["sol_k"] if b else 0.0, "tokens_at_best": b["cum_tokens"] if b else None,
                               "tokens_5_rounds": R[-1]["cum_tokens"], "tokens_lower_bound": R[-1]["cum_tokens_is_lower_bound"],
                               "usd_5_rounds": R[-1]["cum_usd"], "usd_5_rounds_lower_bound": R[-1]["cum_usd_lower_bound"],
                               "api_elapsed_s": sum(r["api_elapsed_s"] for r in R),
                               "statuses": ";".join(r["status"] for r in R), "human_latency_k_ms": h_ms, "human_sol_k": hrow["human_sol_k"],
                               "llm_over_human_sol": (b["sol_k"] / hrow["human_sol_k"]) if (b and hrow["human_sol_k"]) else None})
                hrow[f"{m}_best_latency_k_ms"] = b["latency_k_ms"] if b else None
                hrow[f"{m}_bestsol_k"] = b["sol_k"] if b else 0.0
            human.append(hrow)
    write_csv(OUT / "appendix" / "llm_per_operator.csv", per_op)
    write_csv(OUT / "appendix" / "human_kernel_comparison.csv", human)
    write_csv(OUT / "appendix" / "human_dev_time_template.csv",
              [{"device": d, "dsl": s, "operator": o, "dtype": case_k[o]["dtype"], "human_dev_time_hours": None,
                "source": "missing: no reliable per-operator development-time record in the repository"}
               for d, s in TRACKS for o in eligible[d]])

    # failure breakdown, cost / tokens / time per track, campaign wall time per device
    statuses = sorted({r["status"] for r in rows})
    fail, cost = [], []
    for dev, dsl in TRACKS:
        for m in MODELS:
            R = [r for r in rows if (r["device"], r["dsl"], r["model"]) == (dev, dsl, m)]
            c = Counter(r["status"] for r in R)
            fail.append({"device": dev, "dsl": dsl, "model": m, "rounds": len(R), **{s: c.get(s, 0) for s in statuses},
                         "format_error_truncated_max_tokens": sum(1 for r in R if r["status"] == "format_error" and r["truncated"]),
                         "transport_failed_unknown_charge": sum(r["transport_failed_unknown_charge"] for r in R),
                         "reviewed_rounds": sum(1 for r in R if r["reviews"])})
            api = sorted(r["api_elapsed_s"] for r in R)
            ev = sorted(r["eval_elapsed_s"] for r in R if r["eval_elapsed_s"] is not None)
            cost.append({"device": dev, "dsl": dsl, "model": m, "input_tokens": sum(r["input_tokens"] or 0 for r in R),
                         "output_tokens": sum(r["output_tokens"] or 0 for r in R),
                         "reasoning_tokens": (sum(r["reasoning_tokens"] for r in R) if all(r["reasoning_tokens"] is not None for r in R)
                                              else None),   # Claude reports thinking inside output tokens, not separately
                         "tokens": sum(r["tokens"] or 0 for r in R),
                         "tokens_lower_bound": int(any(r["cum_tokens_is_lower_bound"] for r in R)),
                         "usd_exact": None if any(r["usd_status"] != "exact_from_usage_and_frozen_price" for r in R) else round(sum(r["usd"] for r in R), 2),
                         "usd_lower_bound": round(sum(r["usd_lower_bound"] or 0 for r in R), 2),
                         "attempts_with_unknown_charge": sum(1 for r in R if r["usd_status"] != "exact_from_usage_and_frozen_price"),
                         "api_elapsed_total_h": sum(api) / 3600, "api_elapsed_median_s": api[len(api) // 2],
                         "evaluations": len(ev), "eval_elapsed_total_h": sum(ev) / 3600, "eval_elapsed_median_s": ev[len(ev) // 2] if ev else None,
                         "eval_device_lock_wait_total_h": sum(r["eval_device_lock_wait_s"] or 0 for r in R) / 3600})
    write_csv(OUT / "appendix" / "llm_failure_breakdown.csv", fail)
    write_csv(OUT / "appendix" / "llm_cost_tokens_time.csv", cost)
    wall = []
    for dev in DEVICES:
        R = [r for r in rows if r["device"] == dev]
        t0 = min(r["request_start_t"] for r in R if r["request_start_t"])
        t1 = max(max(r["request_end_t"] or 0, r["eval_end_t"] or 0) for r in R)
        wall.append({"device": dev, "campaign": DEVICES[dev][2], "first_request_start_epoch": round(t0), "last_event_end_epoch": round(t1),
                     "campaign_wall_h": (t1 - t0) / 3600,
                     "note": "first API request start to last request/evaluation end; includes scheduler pauses, review waits and concurrency"})
    write_csv(OUT / "appendix" / "llm_campaign_wall_time.csv", wall)

    paths, layout, curves = plot_curves(g, eligible, OUT)
    with open(OUT / "appendix" / "llm_bestsol_curves.json", "w") as f:
        json.dump(curves, f, indent=1)

    manifest = {
        "script": "scripts/paper_figures/llm_rq3.py", "inputs": {"archives": archives, "frozen": frozen, "human_csv_commit_trn2": MAIN_COMMIT},
        "metric": {"valid": "round passed all 20 frozen cases under its adopted evaluation revision (evaluation_revision)",
                   "case_k": "largest problem_size of the frozen case set of the representative dtype (ties: last)",
                   "sol": "T_SOL,k / T_i,k, frozen empirical scoring table of the device; > 1 kept",
                   "C_i": "cumulative logical input + output tokens of rounds 1..i (failed rounds included); lower bound when a failed "
                          "transport attempt with an unknown charge occurred in rounds <= i",
                   "BestSOL(B)": "max({SOL_i : C_i <= B, valid_i} U {0}); arithmetic mean over eligible operators",
                   "sol_per_million_tokens": "mean BestSOL@5 / (mean tokens per operator / 1e6)",
                   "usd": "frozen price snapshot (api_pricing.yaml, sha256 recorded); unknown charges kept unknown (lower bound reported)"},
        "case_k_equals_representative_case": all(v["matches_representative_case"] for v in case_k.values()),
        "eligible_operators": {d: len(v) for d, v in eligible.items()}, "common_operators": common,
        "rounds": len(rows), "trajectories": len(g), "figure": {"outputs": paths, "layout": layout},
    }
    with open(OUT / "llm_rq3_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(json.dumps({"rounds": len(rows), "layout": layout}, indent=1))


if __name__ == "__main__":
    main()
