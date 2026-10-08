"""QA for the paper figure drafts (CPU only; reads artifacts/paper_figures/{combined,plots}, never the device packages' raw data).

Every plotted number is recomputed here from benchmark_cases_normalized.csv.gz / figure_evidence.csv WITHOUT the
figure_data.Data helper, then compared with the figure manifests and with the text actually rendered into the SVGs.
Writes plots/qa_plots.json; exits 1 if any check fails.

  python scripts/paper_figures/validate_plots.py [--skip-reproducibility]
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PF = REPO / "artifacts" / "paper_figures"
COMBINED, PLOTS = PF / "combined", PF / "plots"
csv.field_size_limit(1 << 30)

FIGURES = {  # name -> (subdir, design width in inches)
    "fig_rq1_cross_accelerator": ("main", 7.0), "fig_rq2_cross_device_diagnosis": ("main", 7.0),
    "fig_rq3_within_device_dsl": ("main", 7.0), "fig_a1_performance_atlas": ("appendix", 7.0),
    "fig_a2_shape_dtype": ("appendix", 7.0), "fig_a3_execution_paths": ("appendix", 7.0),
    "fig_a4_within_device_matrix": ("appendix", 3.35), "fig_a5_profiling_evidence": ("appendix", 7.0)}
ALLOWED_WIDTHS = (3.35, 7.0)
MIN_FONT_PT, MIN_DPI = 5.5, 300
FORMAL_ONLY = ("fig_rq1_cross_accelerator", "fig_rq3_within_device_dsl", "fig_a1_performance_atlas", "fig_a2_shape_dtype",
               "fig_a4_within_device_matrix")
FORMAL_INPUTS = {"benchmark_cases_normalized.csv.gz", "category_mapping.csv", "comparison_manifest.json"}
VENDOR = {"B200": "NVIDIA", "GH200": "NVIDIA", "MI300X": "AMD"}
STATIC_KINDS = ("static", "static_isa", "static_sass", "static_isa+source", "launch_record+static_isa")
SUPPORT = {"B200": ("triton", "cutile", "tilelang"), "GH200": ("triton", "cutile", "tilelang"), "MI300X": ("triton",)}
TOL = 1e-9

results = []


def check(name, ok, detail=""):
    results.append({"check": name, "status": "pass" if ok else "fail", "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def read_csv(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", newline="") as f:
        return list(csv.DictReader(f))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def svg_texts(p):
    body = Path(p).read_text()
    out = []
    for m in re.finditer(r"<text\b([^>]*)>(.*?)</text>", body, re.S):
        out.append((re.sub(r"<[^>]+>", "", m.group(2)).strip(), m.group(1)))
    return out


def svg_font_sizes(p):
    return [float(x) for x in re.findall(r"font-size:\s*([\d.]+)px", Path(p).read_text())]


def pdf_size_in(p):
    m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", Path(p).read_bytes())
    return (float(m.group(3)) - float(m.group(1))) / 72, (float(m.group(4)) - float(m.group(2))) / 72


def png_size_px(p):
    b = Path(p).read_bytes()[:24]
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", b[16:24])


def walk_numbers(o):
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from walk_numbers(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk_numbers(v)


# ------------------------------------------------------------------------------------------- independent recomputation
def load_formal():
    V = defaultdict(dict)       # (dev, dsl, op) -> {case_id_v2: (torch_ms, dsl_ms)}
    params = {}
    for r in read_csv(COMBINED / "benchmark_cases_normalized.csv.gz"):
        if r["mode"] != "autotune" or r["validity"] != "valid":
            continue
        t, d = float(r["torch_ms"]), float(r["dsl_ms"])
        assert t > 0 and d > 0 and math.isfinite(t) and math.isfinite(d)
        V[(r["device"], r["dsl"], r["operator"])][r["case_id_v2"]] = (t, d)
        params[r["case_id_v2"]] = (r["operator"], r["dtype"])
    cats = {r["operator"]: r["category"] for r in read_csv(COMBINED / "category_mapping.csv")}
    return V, cats, params


def S_op(V, dev, dsl, op, ids=None):
    c = V.get((dev, dsl, op), {})
    ids = list(c) if ids is None else [i for i in ids if i in c]
    return gm([c[i][0] / c[i][1] for i in ids]), len(ids)


def matched(V, op, pairs):
    sets = [set(V.get((d, s, op), {})) for d, s in pairs]
    return sorted(set.intersection(*sets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-reproducibility", action="store_true")
    a = ap.parse_args()
    M = {}
    for name, (sub, width) in FIGURES.items():
        mp = PLOTS / "manifests" / f"{name}.json"
        M[name] = json.load(open(mp)) if mp.exists() else None

    # 1. outputs exist and match the manifest hashes
    bad = []
    for name, (sub, _) in FIGURES.items():
        m = M[name]
        if m is None:
            bad.append(f"{name}: manifest missing")
            continue
        for ext, p in (("pdf", PLOTS / sub / f"{name}.pdf"), ("svg", PLOTS / sub / f"{name}.svg"), ("png", PLOTS / "previews" / f"{name}.png")):
            if not p.exists():
                bad.append(f"{name}.{ext} missing")
            elif m["outputs"][ext]["sha256"] != sha(p) or m["outputs"][ext]["path"] != str(p.relative_to(PLOTS)):
                bad.append(f"{name}.{ext} hash/path differs from manifest")
    check("01.outputs_pdf_svg_png_manifest_exist_and_hashes_match", not bad, "; ".join(bad) or f"{len(FIGURES)} figures x 4 files")
    if bad:
        return finish()

    # 2. physical size / resolution / canvas
    bad = []
    for name, (sub, width) in FIGURES.items():
        L = M[name]["layout"]
        w, h = pdf_size_in(PLOTS / sub / f"{name}.pdf")
        if not (abs(w - width) < 0.01 and width in ALLOWED_WIDTHS and abs(h - L["height_in"]) < 0.01):
            bad.append(f"{name}: pdf {w:.3f}x{h:.3f} in, expected width {width}")
        pw, ph = png_size_px(PLOTS / "previews" / f"{name}.png")
        if pw / width < MIN_DPI - 0.5 or ph / L["height_in"] < MIN_DPI - 0.5:
            bad.append(f"{name}: png {pw}x{ph} px < {MIN_DPI} dpi")
        if any(v > 0.0 for v in L["content_overflow_in"].values()):
            bad.append(f"{name}: content outside canvas {L['content_overflow_in']}")
    check("02.width_3.35_or_7.0in_png_300dpi_no_content_outside_canvas", not bad, "; ".join(bad))

    # 3. fonts: rendered sizes (from the SVG files) and embedding (PDF)
    bad = []
    for name, (sub, _) in FIGURES.items():
        sizes = svg_font_sizes(PLOTS / sub / f"{name}.svg")
        if not sizes or min(sizes) < MIN_FONT_PT:
            bad.append(f"{name}: min font {min(sizes) if sizes else None} pt")
        pdf = (PLOTS / sub / f"{name}.pdf").read_bytes()
        if b"/Type3" in pdf or b"CIDFontType2" not in pdf:
            bad.append(f"{name}: PDF fonts not embedded as TrueType (Type 42)")
    check("03.min_font_5.5pt_and_truetype_embedded", not bad, "; ".join(bad))

    # 4. all plotted numbers finite
    bad = [n for n in FIGURES if any(not math.isfinite(x) for x in walk_numbers(M[n]["plotted_values"]))]
    check("04.plotted_values_finite", not bad, ", ".join(bad))

    # 5. source data unchanged since plotting (combined inputs, device packages, results CSVs)
    bad = []
    for n in FIGURES:
        for f, h in M[n]["source_data_files"].items():
            if sha(COMBINED / f) != h:
                bad.append(f"{n}: {f}")
    cm = json.load(open(COMBINED / "comparison_manifest.json"))
    for dev, files in cm["source_packages"].items():
        base = PF / ("amd" if VENDOR[dev] == "AMD" else "nvidia") / dev
        for f, h in files.items():
            if not (base / f).exists() or sha(base / f) != h:
                bad.append(f"package {dev}/{f}")
    for f, h in cm["source_csv_sha256"].items():
        if sha(REPO / f) != h:
            bad.append(f)
    st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "artifacts/paper_figures/nvidia", "artifacts/paper_figures/amd",
                         "results"], capture_output=True, text=True).stdout.strip()
    if st:
        bad.append("uncommitted changes under device packages/results: " + st.replace("\n", " | "))
    check("05.source_data_unchanged", not bad, "; ".join(bad[:8]))

    V, cats, params = load_formal()
    ops = sorted(cats)

    # 6. formal-latency figures use only the formal benchmark CSV (no profiler duration, no diagnostic latency)
    bad = []
    for n in FORMAL_ONLY:
        if M[n]["profiling_evidence_ids"] or not set(M[n]["source_data_files"]) <= FORMAL_INPUTS:
            bad.append(n)
    ev = {r["evidence_id"]: r for r in read_csv(COMBINED / "figure_evidence.csv")}
    profile_ids = {r["profile_id"] for r in read_csv(COMBINED / "profile_index_normalized.csv")}
    EVIDENCE_FIGS = ("fig_rq2_cross_device_diagnosis", "fig_a5_profiling_evidence")   # A3 cites profile_ids instead
    a5_axes = M["fig_a5_profiling_evidence"]["axes_evidence"]
    dur = [i for ids in a5_axes.values() for i in ids if "time_duration" in ev[i]["metric_name"]]
    diag_outside = [i for n in EVIDENCE_FIGS if n != "fig_a5_profiling_evidence" for i in M[n]["profiling_evidence_ids"]
                    if ev[i]["measurement_kind"] == "diagnostic_experiment"]
    diag_wrong_axis = [k for k, ids in a5_axes.items() if any(ev[i]["measurement_kind"] == "diagnostic_experiment" for i in ids) and "diagnostic" not in k]
    a5_txt = " ".join(t for t, _ in svg_texts(PLOTS / "appendix" / "fig_a5_profiling_evidence.svg"))
    check("06.no_profiler_or_diagnostic_latency_used_as_formal_latency",
          not (bad or dur or diag_outside or diag_wrong_axis) and a5_txt.count("diagnostic run") >= 2,
          f"formal-only violations {bad}; profiler durations {dur}; diagnostic outside A5 {diag_outside}; unlabeled diagnostic axes {diag_wrong_axis}")

    # 7. RQ1 recomputed + rendered labels + N/A cells (never zero)
    m = M["fig_rq1_cross_accelerator"]
    bad, labels = [], Counter()
    for c in m["plotted_values"]:
        rows_ops = ops if c["row"] == "Overall" else [o for o in ops if cats[o] == c["row"]]
        if c["dsl"] not in SUPPORT[c["device"]]:
            exp = None
        else:
            exp = gm([S_op(V, c["device"], c["dsl"], o)[0] for o in rows_ops])
        if not close(exp, c["speedup"]):
            bad.append(f"{c['row']}/{c['device']}/{c['dsl']}: {c['speedup']} vs {exp}")
        if exp is not None:
            labels[f"{exp:.2f}"] += 1
    t = Counter(x for x, _ in svg_texts(PLOTS / "main" / "fig_rq1_cross_accelerator.svg"))
    miss = labels - t
    na = sum(1 for c in m["plotted_values"] if c["speedup"] is None)
    check("07.rq1_values_recomputed_labels_match_na_not_zero", not bad and not miss and t["N/A"] == na == 12,
          f"value mismatches {bad[:3]}; labels missing in SVG {dict(miss)}; N/A cells {na}, N/A texts {t['N/A']}")

    # 8. RQ3 recomputed (matched intersections, winners) + rendered winner text
    m = M["fig_rq3_within_device_dsl"]
    bad = []
    win = {}
    for p in m["plotted_values"]:
        dev, op = p["device"], p["operator"]
        ids = matched(V, op, [(dev, "triton"), (dev, "cutile"), (dev, "tilelang")])
        lat = {s: gm([V[(dev, s, op)][i][1] for i in ids]) for s in ("triton", "cutile", "tilelang")}
        w = min(lat, key=lat.get)
        ru = sorted(lat.values())[1] / lat[w]
        win.setdefault(dev, Counter())[w] += 1
        win.setdefault(dev + "_near", Counter())[w] += ru <= 1.05
        if not (len(ids) == p["n_cases"] > 0 and close(lat["cutile"] / lat["triton"], p["x_cutile_over_triton"])
                and close(lat["tilelang"] / lat["triton"], p["y_tilelang_over_triton"]) and w == p["winner"]):
            bad.append(f"{dev}/{op}")
    txt = " ".join(x for x, _ in svg_texts(PLOTS / "main" / "fig_rq3_within_device_dsl.svg"))
    for dev in ("B200", "GH200"):
        c, n = win[dev], win[dev + "_near"]
        if f"Triton {c['triton']} · TileLang {c['tilelang']} · cuTile {c['cutile']}" not in txt or \
                f"Triton {n['triton']} · TileLang {n['tilelang']} · cuTile {n['cutile']}" not in txt:
            bad.append(f"{dev} winner text")
        if dict(c) != {k: v for k, v in m["winners"][dev]["counts"].items() if v}:
            bad.append(f"{dev} winner counts")
    check("08.rq3_values_recomputed_over_explicit_3dsl_intersection_winner_text_matches", not bad,
          "; ".join(bad[:5]) or "B200 " + str(dict(win["B200"])) + ", GH200 " + str(dict(win["GH200"])))

    # 9. A1 recomputed: per-op speedups and matched-case deltas (+ rendered labels)
    m = M["fig_a1_performance_atlas"]
    bad, labels = [], Counter()
    for c in m["plotted_values"]["left"]:
        exp = S_op(V, c["device"], c["dsl"], c["operator"])[0]
        if not close(exp, c["speedup"]):
            bad.append(f"S {c}")
        if exp is not None:
            labels[f"{exp:.2f}"] += 1
    nm = m["case_coverage"]["right_matched_cases"]
    for c in m["plotted_values"]["right"]:
        ids = matched(V, c["operator"], [(c["from"], c["dsl"]), (c["to"], c["dsl"])])
        exp = math.log2(S_op(V, c["to"], c["dsl"], c["operator"], ids)[0] / S_op(V, c["from"], c["dsl"], c["operator"], ids)[0]) if ids else None
        if not close(exp, c["delta_log2"]) or nm[f"{c['operator']}|{c['dsl']}|{c['to']}/{c['from']}"] != len(ids):
            bad.append(f"Δ {c}")
        if exp is not None:
            labels[f"{2 ** exp:.2f}"] += 1
    t = Counter(x for x, _ in svg_texts(PLOTS / "appendix" / "fig_a1_performance_atlas.svg"))
    miss = labels - t
    na_exp = sum(1 for c in m["plotted_values"]["left"] if c["speedup"] is None) + sum(1 for c in m["plotted_values"]["right"] if c["delta_log2"] is None)
    check("09.a1_values_recomputed_deltas_over_matched_cases_labels_match", not bad and not miss and t["N/A"] == na_exp,
          f"mismatches {bad[:3]}; missing labels {dict(list(miss.items())[:5])}; N/A {t['N/A']} vs {na_exp}")

    # 10. A2 per-case values + A4 slowdowns recomputed
    bad = []
    for c in M["fig_a2_shape_dtype"]["plotted_values"]:
        tv = V[(c["device"], c["dsl"], c["operator"])].get(c["case_id_v2"])
        if tv is None or not close(tv[0] / tv[1], c["speedup"]) or params[c["case_id_v2"]] != (c["operator"], c["dtype"]):
            bad.append(f"A2 {c['operator']}/{c['dtype']}/{c['device']}/{c['dsl']}")
    m = M["fig_a4_within_device_matrix"]
    for dev in ("B200", "GH200"):
        for op in ops:
            ids = matched(V, op, [(dev, s) for s in SUPPORT[dev]])
            lat = {s: gm([V[(dev, s, op)][i][1] for i in ids]) for s in SUPPORT[dev]}
            best = min(lat.values())
            if m["case_coverage"][f"{op}|{dev}"] != len(ids):
                bad.append(f"A4 coverage {op}|{dev}")
            for c in m["plotted_values"]:
                if c["operator"] == op and c["device"] == dev and not close(lat[c["dsl"]] / best, c["slowdown"]):
                    bad.append(f"A4 {op}/{dev}/{c['dsl']}")
    check("10.a2_case_values_and_a4_slowdowns_recomputed", not bad, "; ".join(bad[:5]))

    # 11. RQ2 panel A: formal latency at ONE case_id_v2 per case (same on every device, = the profiled case)
    m = M["fig_rq2_cross_device_diagnosis"]
    bad = []
    by = defaultdict(set)
    prof = defaultdict(set)
    for r in read_csv(COMBINED / "profile_index_normalized.csv"):
        prof[(r["device"], r["dsl"], r["operator"], r["dtype"])].add(r["case_id_v2"])
    for c in m["plotted_values"]:
        op, dt = c["case"].split("/")
        if c["dsl"] not in SUPPORT[c["device"]]:
            continue
        by[c["case"]].add(c["case_id_v2"])
        tv = V[(c["device"], c["dsl"], op)].get(c["case_id_v2"])
        if c["speedup"] is None:
            if tv is not None:
                bad.append(f"{c['case']}/{c['device']}/{c['dsl']} has data but plotted as missing")
            continue
        if tv is None or not close(tv[0] / tv[1], c["speedup"]) or tv != (c["torch_ms"], c["dsl_ms"]):
            bad.append(f"{c['case']}/{c['device']}/{c['dsl']}")
        if c["case_id_v2"] not in prof[(c["device"], c["dsl"], op, dt)]:
            bad.append(f"{c['case']}/{c['device']}/{c['dsl']} not the profiled case")
    multi = [k for k, v in by.items() if len(v) != 1]
    check("11.rq2_formal_speedup_at_single_profiled_case_id_v2", not bad and not multi, f"{bad[:4]} multi-id cases {multi}")

    # 12. evidence values equal figure_evidence.csv; A5 axes vendor-homogeneous and never mix static with dynamic
    bad = []
    for n in EVIDENCE_FIGS:
        for i in M[n]["profiling_evidence_ids"]:
            if i not in ev:
                bad.append(f"{n}: unknown evidence {i}")
    bad += [f"A3: unknown profile {i}" for i in M["fig_a3_execution_paths"]["profiling_evidence_ids"] if i not in profile_ids]
    for r in M["fig_a5_profiling_evidence"]["plotted_values"]:
        if ev[r["evidence_id"]] != r:
            bad.append(f"A5 value differs {r['evidence_id']}")
    for axis, ids in a5_axes.items():
        vend = {VENDOR[ev[i]["device"]] for i in ids}
        kinds = {ev[i]["measurement_kind"] for i in ids}
        if len(vend) != 1:
            bad.append(f"{axis}: vendors {vend}")
        if kinds & set(STATIC_KINDS) and kinds - set(STATIC_KINDS):
            bad.append(f"{axis}: static+dynamic {kinds}")
    ax_a = [ev[i] for i in a5_axes["A_nvidia_dynamic_instruction_ratio"]]
    if {(r["metric_name"], r["measurement_kind"]) for r in ax_a} != {("smsp__inst_executed.sum", "ncu_counter")}:
        bad.append("A5 panel A ratio built from different counters")
    check("12.evidence_traceable_no_cross_vendor_axis_no_static_dynamic_mix", not bad, "; ".join(bad[:5]) or
          ", ".join(f"{k}:{sorted({VENDOR[ev[i]['device']] for i in v})}" for k, v in a5_axes.items()))

    # 13. missing evidence drawn as 'n/c', never as a zero bar
    a5 = M["fig_a5_profiling_evidence"]
    axis_rows = [ev[i] for ids in a5_axes.values() for i in ids]
    missing = [r for r in axis_rows if r["value"] in ("", None)]
    zeros = [r["evidence_id"] for r in axis_rows if r["value"] not in ("", None) and float(r["value"]) == 0.0]
    nc = sum(1 for x, _ in svg_texts(PLOTS / "appendix" / "fig_a5_profiling_evidence.svg") if x == "n/c")
    check("13.missing_evidence_shown_as_nc_not_zero", nc >= len(missing) > 0,
          f"{len(missing)} missing evidence rows, {nc} 'n/c' labels; genuine measured zeros: {zeros}")

    # 14. A3 cells reproduce execution_path_matrix.csv; static-only cells are listed as such
    rows = read_csv(COMBINED / "execution_path_matrix.csv")
    a3 = M["fig_a3_execution_paths"]["plotted_values"]
    st = [r for r in a3 if r["evidence_kind"] in STATIC_KINDS]
    check("14.a3_cells_equal_execution_path_matrix", a3 == rows, f"{len(a3)} cells, {len(st)} static-only (dashed)")

    # 15. scope: no NKI, no RQ4 numbers, no placeholder figures
    bad = []
    for n in FIGURES:
        txt = " ".join(x for x, _ in svg_texts(PLOTS / FIGURES[n][0] / f"{n}.svg"))
        if re.search(r"\bNKI\b|Trainium|trn2", txt) or re.search(r'"(nki|trn2)"', json.dumps(M[n]["plotted_values"])):
            bad.append(n)
    extra = sorted(p.name for p in PLOTS.rglob("*") if p.is_file() and p.suffix in (".pdf", ".svg", ".png") and p.stem not in FIGURES)
    top = PLOTS / "plot_manifest.json"
    todo_ok = True
    if top.exists():
        tm = json.load(open(top))
        todo = [f for f in tm["figures"] if f.get("status") == "todo"]
        todo_ok = {f["id"] for f in todo} == {"fig5_rq4_llm", "appendix_llm"} and all("outputs" not in f for f in todo)
        for f in tm["figures"]:
            if f.get("status") == "generated":
                for ext, o in f["outputs"].items():
                    if sha(PLOTS / o["path"]) != o["sha256"]:
                        todo_ok = False
    check("15.no_nki_no_rq4_numbers_no_extra_figures_top_manifest_consistent", not bad and not extra and todo_ok,
          f"NKI in {bad}; unexpected files {extra}; top manifest {'ok' if todo_ok else 'inconsistent'}")

    # 16. reproducibility: regenerate everything into a temp dir and compare hashes
    if a.skip_reproducibility:
        results.append({"check": "16.reproducible_byte_identical", "status": "skipped", "detail": "--skip-reproducibility"})
        print("[SKIP] 16.reproducible_byte_identical")
    else:
        with tempfile.TemporaryDirectory() as td:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{REPO}:{HERE}", MPLBACKEND="Agg")
            py = sys.executable
            subprocess.run([py, str(HERE / "build_figure_evidence.py"), "--repo", str(REPO), "--out", f"{td}/combined"], check=True,
                           env=env, capture_output=True)
            code = ("import sys; sys.path.insert(0, %r); import plot_rq1, plot_rq2, plot_rq3, plot_appendix; o = %r; "
                    "plot_rq1.main(o); plot_rq2.main(o); plot_rq3.main(o); plot_appendix.main(o)") % (str(HERE), f"{td}/plots")
            subprocess.run([py, "-c", code], check=True, env=env, capture_output=True)
            diff = [f for f in ("figure_evidence.csv", "execution_path_matrix.csv", "rq2_case_selection.json")
                    if sha(f"{td}/combined/{f}") != sha(COMBINED / f)]
            for name, (sub, _) in FIGURES.items():
                for rel in (f"{sub}/{name}.pdf", f"{sub}/{name}.svg", f"previews/{name}.png"):
                    if sha(f"{td}/plots/{rel}") != sha(PLOTS / rel):
                        diff.append(rel)
                a_, b_ = json.load(open(f"{td}/plots/manifests/{name}.json")), M[name]
                if a_["plotted_values"] != b_["plotted_values"]:
                    diff.append(f"{name}.json plotted_values")
        check("16.reproducible_byte_identical", not diff, f"differs: {diff}" if diff else "3 evidence files + 24 figure files + plotted values identical")
    return finish()


def finish():
    ok = all(r["status"] != "fail" for r in results)
    out = {"gate": "pass" if ok else "fail", "checks": results,
           "scope": "CPU-only checks on artifacts/paper_figures/plots; no GPU benchmark, NCU or ROCm run"}
    json.dump(out, open(PLOTS / "qa_plots.json", "w"), indent=1)
    print("GATE:", out["gate"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
