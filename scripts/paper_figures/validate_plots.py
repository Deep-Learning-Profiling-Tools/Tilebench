"""QA for the paper figure drafts (CPU only; reads artifacts/paper_figures/{combined,plots} and, for counter attribution, the
NVIDIA/AMD packages' derived tables; never a raw report).

Every plotted number is recomputed here from benchmark_cases_normalized.csv.gz / figure_evidence.csv WITHOUT the
figure_data.Data helper, then compared with the figure manifests and with the text rendered into the SVGs.
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

ACL_TEXT_IN, ACL_COLUMN_IN = 6.30, 3.03          # acl.sty: A4, 2.5 cm margins, 0.6 cm column separation
FIGURES = {  # name -> (subdir, design width in inches, minimum font in pt at that width)
    "fig_rq1_cross_accelerator": ("main", ACL_TEXT_IN, 7.0), "fig_rq2_cross_device_diagnosis": ("main", ACL_TEXT_IN, 7.0),
    "fig_rq3_within_device_dsl": ("main", ACL_TEXT_IN, 7.0), "fig_a1_performance_atlas": ("appendix", ACL_TEXT_IN, 6.0),
    "fig_a2_shape_dtype": ("appendix", ACL_TEXT_IN, 6.0), "fig_a3_execution_paths": ("appendix", ACL_TEXT_IN, 6.0),
    "fig_a4_within_device_matrix": ("appendix", ACL_TEXT_IN, 6.0), "fig_a5_profiling_evidence": ("appendix", ACL_TEXT_IN, 6.0)}
MIN_DPI = 300
FORMAL_ONLY = ("fig_rq1_cross_accelerator", "fig_rq3_within_device_dsl", "fig_a1_performance_atlas", "fig_a2_shape_dtype",
               "fig_a4_within_device_matrix")
FORMAL_INPUTS = {"benchmark_cases_normalized.csv.gz", "category_mapping.csv", "comparison_manifest.json"}
VENDOR = {"B200": "NVIDIA", "GH200": "NVIDIA", "MI300X": "AMD"}
STATIC_KINDS = ("static", "static_isa", "static_sass", "static_isa+source", "launch_record+static_isa")
DYNAMIC_NV_KINDS = ("ncu_counter", "dynamic_sass_count")
SUPPORT = {"B200": ("triton", "cutile", "tilelang"), "GH200": ("triton", "cutile", "tilelang"), "MI300X": ("triton",)}
RQ2_COUNTERS = {  # Figure 3 axis label -> (evidence metric, divisor)
    "TMA load bytes (GB)": ("l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum", 1e9),
    "Shared-memory stores (M)": ("sass__inst_executed_per_opcode_with_modifier_all[STS*]", 1e6),
    "Executed instructions (M)": ("smsp__inst_executed.sum", 1e6),
    "Global store instructions (M)": ("sass__inst_executed_per_opcode_with_modifier_all[STG*]", 1e6),
    "L1 global-load sectors (M)": ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", 1e6),
    "Achieved occupancy (%)": ("sm__warps_active.avg.pct_of_peak_sustained_active", 1.0)}
SCRIPTS = ["plot_style.py", "figure_data.py", "plot_rq1.py", "plot_rq2.py", "plot_rq3.py", "plot_appendix.py"]
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
    return [re.sub(r"<[^>]+>", "", m.group(2)).strip() for m in re.finditer(r"<text\b([^>]*)>(.*?)</text>", body, re.S)]


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
    return sorted(set.intersection(*[set(V.get((d, s, op), {})) for d, s in pairs]))


def winners(V, ops):
    out = {}
    for dev in ("B200", "GH200"):
        cnt, near = Counter(), Counter()
        for op in ops:
            ids = matched(V, op, [(dev, s) for s in SUPPORT[dev]])
            lat = {s: gm([V[(dev, s, op)][i][1] for i in ids]) for s in SUPPORT[dev]}
            w = min(lat, key=lat.get)
            cnt[w] += 1
            near[w] += sorted(lat.values())[1] / lat[w] <= 1.05
        out[dev] = (cnt, near)
    return out


def build_into(td):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{REPO}:{HERE}", MPLBACKEND="Agg")
    subprocess.run([sys.executable, str(HERE / "build_figure_evidence.py"), "--repo", str(REPO), "--out", f"{td}/combined"], check=True,
                   env=env, capture_output=True)
    code = ("import sys; sys.path.insert(0, %r); import plot_rq1, plot_rq2, plot_rq3, plot_appendix; o = %r; "
            "plot_rq1.main(o); plot_rq2.main(o); plot_rq3.main(o); plot_appendix.main(o)") % (str(HERE), f"{td}/plots")
    subprocess.run([sys.executable, "-c", code], check=True, env=env, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-reproducibility", action="store_true")
    a = ap.parse_args()
    M = {}
    for name, (sub, _, _) in FIGURES.items():
        mp = PLOTS / "manifests" / f"{name}.json"
        M[name] = json.load(open(mp)) if mp.exists() else None

    # 01. outputs exist and match the manifest hashes
    bad = []
    for name, (sub, _, _) in FIGURES.items():
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

    # 02. regenerated by the current code (manifest code hashes == current scripts)
    cur = {f"scripts/paper_figures/{f}": sha(HERE / f) for f in SCRIPTS}
    bad = [f"{n}: {k}" for n in FIGURES for k, h in M[n]["code_sha256"].items() if cur.get(k, sha(REPO / k)) != h]
    check("02.every_output_regenerated_by_current_code", not bad, "; ".join(bad[:5]) or "all manifests pin the current plotting code")

    # 03. final ACL dimensions, resolution, canvas
    bad = []
    for name, (sub, width, _) in FIGURES.items():
        L = M[name]["layout"]
        w, h = pdf_size_in(PLOTS / sub / f"{name}.pdf")
        if not (abs(w - width) < 0.005 and abs(h - L["height_in"]) < 0.005):
            bad.append(f"{name}: pdf {w:.3f}x{h:.3f} in, expected width {width}")
        pw, ph = png_size_px(PLOTS / "previews" / f"{name}.png")
        if pw / width < MIN_DPI - 0.5 or ph / L["height_in"] < MIN_DPI - 0.5:
            bad.append(f"{name}: png {pw}x{ph} px < {MIN_DPI} dpi")
        if any(v > 0.0 for v in L["content_overflow_in"].values()):
            bad.append(f"{name}: content outside canvas {L['content_overflow_in']}")
    check("03.acl_width_6.30in_png_300dpi_no_content_outside_canvas", not bad, "; ".join(bad))

    # 04. fonts at the final printed size (figures are drawn at the ACL width, so no scaling) and embedding
    bad, mins = [], {}
    for name, (sub, _, fmin) in FIGURES.items():
        sizes = svg_font_sizes(PLOTS / sub / f"{name}.svg")
        mins[name] = min(sizes)
        if min(sizes) < fmin:
            bad.append(f"{name}: min font {min(sizes)} pt < {fmin}")
        pdf = (PLOTS / sub / f"{name}.pdf").read_bytes()
        if b"/Type3" in pdf or b"CIDFontType2" not in pdf:
            bad.append(f"{name}: PDF fonts not embedded as TrueType (Type 42)")
    check("04.fonts_main>=7pt_appendix>=6pt_truetype_embedded", not bad, "; ".join(bad) or json.dumps(mins))

    # 05. no text overlaps (rendered text boxes, recorded at save time; re-derived by the rebuilds in check 22)
    bad = [f"{n}: {M[n]['layout']['text_overlaps'][:3]}" for n in FIGURES if M[n]["layout"]["text_overlaps"]]
    check("05.no_text_label_overlap", not bad, "; ".join(bad))

    # 06. all plotted numbers finite
    bad = [n for n in FIGURES if any(not math.isfinite(x) for x in walk_numbers(M[n]["plotted_values"]))]
    check("06.plotted_values_finite", not bad, ", ".join(bad))

    # 07. original extraction artifacts and inputs unchanged
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
    check("07.original_extraction_artifacts_unchanged", not bad, "; ".join(bad[:8]))

    V, cats, params = load_formal()
    ops = sorted(cats)
    ev = {r["evidence_id"]: r for r in read_csv(COMBINED / "figure_evidence.csv")}
    profile_ids = {r["profile_id"] for r in read_csv(COMBINED / "profile_index_normalized.csv")}
    rq2, a5 = M["fig_rq2_cross_device_diagnosis"], M["fig_a5_profiling_evidence"]
    a5_axes = a5["axes_evidence"]

    # 08. formal latency only (no profiler duration, no diagnostic latency) in performance plots
    bad = [n for n in FORMAL_ONLY if M[n]["profiling_evidence_ids"] or not set(M[n]["source_data_files"]) <= FORMAL_INPUTS]
    dur = [i for n in ("fig_rq2_cross_device_diagnosis", "fig_a5_profiling_evidence") for i in M[n]["profiling_evidence_ids"]
           if "time_duration" in ev[i]["metric_name"]]
    check("08.no_profiler_duration_as_benchmark_latency", not bad and not dur, f"formal-only violations {bad}; profiler durations used {dur}")

    # 09. RQ1 aggregation recomputed; exactly the seven supported columns; rendered labels
    m = M["fig_rq1_cross_accelerator"]
    bad, labels = [], Counter()
    cols = {(c["device"], c["dsl"]) for c in m["plotted_values"]}
    if cols != {(d, s) for d in SUPPORT for s in SUPPORT[d]}:
        bad.append(f"columns {sorted(cols)}")
    for c in m["plotted_values"]:
        rows_ops = ops if c["row"] == "Overall" else [o for o in ops if cats[o] == c["row"]]
        exp = gm([S_op(V, c["device"], c["dsl"], o)[0] for o in rows_ops])
        if not close(exp, c["speedup"]) or c["n_operators"] != len(rows_ops):
            bad.append(f"{c['row']}/{c['device']}/{c['dsl']}")
        labels[f"{exp:.2f}"] += 1
    miss = labels - Counter(svg_texts(PLOTS / "main" / "fig_rq1_cross_accelerator.svg"))
    check("09.rq1_operator_balanced_gm_recomputed_7_columns_labels_match", not bad and not miss, f"mismatches {bad[:3]}; labels missing {dict(miss)}")

    # 10. RQ3 + A4: per-operator winners over the explicit three-DSL intersection; rendered winner text
    m = M["fig_rq3_within_device_dsl"]
    W_ = winners(V, ops)
    bad = []
    for p in m["plotted_values"]:
        dev, op = p["device"], p["operator"]
        ids = matched(V, op, [(dev, s) for s in SUPPORT[dev]])
        lat = {s: gm([V[(dev, s, op)][i][1] for i in ids]) for s in SUPPORT[dev]}
        if not (len(ids) == p["n_cases"] > 0 and close(lat["cutile"] / lat["triton"], p["x_cutile_over_triton"])
                and close(lat["tilelang"] / lat["triton"], p["y_tilelang_over_triton"]) and min(lat, key=lat.get) == p["winner"]):
            bad.append(f"{dev}/{op}")
    txt = " ".join(svg_texts(PLOTS / "main" / "fig_rq3_within_device_dsl.svg"))
    for dev, (cnt, near) in W_.items():
        if f"fastest: Triton {cnt['triton']}, TileLang {cnt['tilelang']}, cuTile {cnt['cutile']}" not in txt:
            bad.append(f"{dev} winner text")
        if any(m["winners"][dev]["counts"][s] != cnt[s] or m["winners"][dev]["winner_within_5pct"][s] != near[s] for s in SUPPORT[dev]):
            bad.append(f"{dev} winner counts in manifest")
    for c in M["fig_a4_within_device_matrix"]["plotted_values"]:
        dev, op = c["device"], c["operator"]
        ids = matched(V, op, [(dev, s) for s in SUPPORT[dev]])
        lat = {s: gm([V[(dev, s, op)][i][1] for i in ids]) for s in SUPPORT[dev]}
        if not close(lat[c["dsl"]] / min(lat.values()), c["slowdown"]):
            bad.append(f"A4 {dev}/{op}/{c['dsl']}")
    check("10.per_operator_dsl_winners_correct_rq3_a4", not bad,
          "; ".join(bad[:5]) or " | ".join(f"{d}: {dict(c)} (within 5%: {dict(n)})" for d, (c, n) in W_.items()))

    # 11. A1 deltas over matched cases + A2 per-case values
    m = M["fig_a1_performance_atlas"]
    bad, labels = [], Counter()
    for c in m["plotted_values"]["left"]:
        exp = S_op(V, c["device"], c["dsl"], c["operator"])[0]
        if not close(exp, c["speedup"]):
            bad.append(f"S {c['operator']}/{c['device']}/{c['dsl']}")
        labels[f"{exp:.2f}"] += 1
    nm = m["case_coverage"]["right_matched_cases"]
    for c in m["plotted_values"]["right"]:
        ids = matched(V, c["operator"], [(c["from"], c["dsl"]), (c["to"], c["dsl"])])
        exp = math.log2(S_op(V, c["to"], c["dsl"], c["operator"], ids)[0] / S_op(V, c["from"], c["dsl"], c["operator"], ids)[0])
        if not close(exp, c["delta_log2"]) or nm[f"{c['operator']}|{c['dsl']}|{c['to']}/{c['from']}"] != len(ids):
            bad.append(f"Δ {c['operator']}/{c['dsl']}/{c['to']}")
        labels[f"{2 ** exp:.2f}"] += 1
    miss = labels - Counter(svg_texts(PLOTS / "appendix" / "fig_a1_performance_atlas.svg"))
    for c in M["fig_a2_shape_dtype"]["plotted_values"]:
        tv = V[(c["device"], c["dsl"], c["operator"])].get(c["case_id_v2"])
        if tv is None or not close(tv[0] / tv[1], c["speedup"]) or params[c["case_id_v2"]] != (c["operator"], c["dtype"]):
            bad.append(f"A2 {c['operator']}/{c['dtype']}/{c['device']}/{c['dsl']}")
    check("11.cross_device_deltas_over_matched_case_id_v2_a1_a2", not bad and not miss, f"{bad[:4]}; missing labels {dict(list(miss.items())[:4])}")

    # 12. Figure 3: one profiled case_id_v2 per case on every device; formal latency at that case
    bad = []
    prof = defaultdict(set)
    for r in read_csv(COMBINED / "profile_index_normalized.csv"):
        prof[(r["device"], r["dsl"], r["operator"], r["dtype"])].add(r["case_id_v2"])
    by = defaultdict(set)
    for c in rq2["plotted_values"]["speedup"]:
        op, dt = c["case"].split("/")
        by[c["case"]].add(c["case_id_v2"])
        tv = V[(c["device"], c["dsl"], op)].get(c["case_id_v2"])
        if tv is None or tv != (c["torch_ms"], c["dsl_ms"]) or not close(tv[0] / tv[1], c["speedup"]):
            bad.append(f"{c['case']}/{c['device']}/{c['dsl']} latency")
        if c["case_id_v2"] not in prof[(c["device"], c["dsl"], op, dt)]:
            bad.append(f"{c['case']}/{c['device']}/{c['dsl']} not the profiled case")
    multi = [k for k, v in by.items() if len(v) != 1]
    sel = json.load(open(COMBINED / "rq2_case_selection.json"))
    if [s["case"] for s in sel["selected"]] != ["matmul_fp32_fp16_fp8/fp32", "destindex/int8", "1d_conv/fp16"] or len(by) != 3:
        bad.append("Figure 3 must show exactly the three mechanism cases")
    check("12.fig3_exact_profiled_case_formal_latency", not bad and not multi, f"{bad[:4]} multi-id {multi}")

    # 13. device-native counter units and denominators (Figure 3 bars and A5 panel A) recomputed from figure_evidence.csv
    bad = []
    for case, metrics in rq2["plotted_values"]["counters"].items():
        for lab, vals in metrics.items():
            if lab == "speedup":
                continue
            metric, div = RQ2_COUNTERS[lab]
            for k, v in vals.items():
                dev, dsl = k.split(":")
                r = ev[f"rq2:{case}:{dev}:{dsl}:{metric}"]
                if VENDOR[dev] != "NVIDIA" or r["measurement_kind"] not in DYNAMIC_NV_KINDS or not close(float(r["value"]) / div, v):
                    bad.append(f"{case} {lab} {k}")
    ax_a = [ev[i] for i in a5_axes["A_nvidia_dynamic_instruction_ratio"]]
    if {(r["metric_name"], r["measurement_kind"]) for r in ax_a} != {("smsp__inst_executed.sum", "ncu_counter")}:
        bad.append("A5 panel A ratio built from different counters")
    check("13.counter_units_and_denominators_recomputed", not bad, "; ".join(bad[:5]) or "Figure 3 bars and A5 panel A")

    # 14. no cross-vendor numeric axis; static and dynamic counts never share an axis; evidence traceable
    bad = []
    for n in ("fig_rq2_cross_device_diagnosis", "fig_a5_profiling_evidence"):
        bad += [f"{n}: unknown evidence {i}" for i in M[n]["profiling_evidence_ids"] if i not in ev]
    bad += [f"A3: unknown profile {i}" for i in M["fig_a3_execution_paths"]["profiling_evidence_ids"] if i not in profile_ids]
    for r in a5["plotted_values"]:
        if ev[r["evidence_id"]] != r:
            bad.append(f"A5 value differs {r['evidence_id']}")
    for axis, ids in a5_axes.items():
        vend = {VENDOR[ev[i]["device"]] for i in ids}
        kinds = {ev[i]["measurement_kind"] for i in ids}
        if len(vend) != 1:
            bad.append(f"{axis}: vendors {vend}")
        if kinds & set(STATIC_KINDS) and kinds - set(STATIC_KINDS):
            bad.append(f"{axis}: static+dynamic {kinds}")
    if M["fig_a3_execution_paths"]["plotted_values"] != read_csv(COMBINED / "execution_path_matrix.csv"):
        bad.append("A3 cells differ from execution_path_matrix.csv")
    check("14.no_cross_vendor_axis_static_dynamic_distinct", not bad, "; ".join(bad[:5]) or
          ", ".join(f"{k}:{sorted({VENDOR[ev[i]['device']] for i in v})}" for k, v in a5_axes.items()))

    # 15. MI300X diagnostic latency only in A5, on axes and titles labelled 'diagnostic'
    diag_out = [i for i in rq2["profiling_evidence_ids"] if ev[i]["measurement_kind"] == "diagnostic_experiment"]
    diag_axes = [k for k, ids in a5_axes.items() if any(ev[i]["measurement_kind"] == "diagnostic_experiment" for i in ids)]
    t5 = svg_texts(PLOTS / "appendix" / "fig_a5_profiling_evidence.svg")
    ok = not diag_out and all("diagnostic" in k for k in diag_axes) and sum("(diagnostic)" in t for t in t5) >= len(diag_axes) == 2
    check("15.mi300x_diagnostic_latency_labelled_separately", ok, f"diagnostic axes {diag_axes}; in Figure 3: {diag_out}")

    # 16. cache-modifier names: Triton modifiers vs emitted gfx942 bits, from the source experiment records
    exp = {r["variant"]: r for r in read_csv(PF / "amd" / "MI300X" / "diagnostic_experiments.csv")
           if r["experiment_id"] == "cache_modifier_ablation_fp32"}

    def isa(v):
        ld, stv = exp[v]["other_configuration_changes"].split("emitted VMEM: ")[1].split(" | ")
        return ld.strip(), stv.strip()
    bad = []
    if not isa("load.cg_store(default)")[0].endswith("sc0 nt") or isa("load.cg_store(default)")[1].endswith("nt"):
        bad.append(".cg load must emit 'sc0 nt' on loads only")
    if not isa("load(default)_store.cs")[1].endswith("sc0 nt") or isa("load(default)_store.cs")[0].endswith("nt"):
        bad.append(".cs store must emit 'sc0 nt' on stores only")
    if isa("load(default)_store.cg") != isa("load(default)_store(default)"):
        bad.append(".cg store should leave the ISA unchanged")
    conf = a5["confounders"]["MI300X vector_add"]
    for s_ in (".cs -> sc0 nt", ".wt -> sc0 sc1", ".cg -> no ISA change"):
        if s_ not in conf:
            bad.append(f"manifest lacks '{s_}'")
    if any(re.search(r"\bst\.cg\b|\bld\.cs\b", t) for t in t5):
        bad.append("figure text names a store .cg / load .cs modifier")
    if not any(".cg loads" in t for t in t5):
        bad.append("A5 legend lacks '.cg loads'")
    check("16.cache_modifier_names_match_emitted_isa", not bad, "; ".join(bad) or "ld.cg -> loads sc0 nt; st.cs -> stores sc0 nt; st.cg -> unchanged")

    # 17. STS/WGMMA attribution: same profile, single launch, both dynamic warp-level; recomputed from instruction_mix.csv
    bad = []
    mix = defaultdict(list)
    for r in read_csv(PF / "nvidia" / "GH200" / "instruction_mix.csv"):
        mix[r["profile_id"]].append(r)
    for i in a5_axes["B_gh200_sts_per_wgmma"]:
        r = ev[i]
        rows = mix[r["profile_id"]]
        if len({x["launch_id"] for x in rows}) != 1:
            bad.append(f"{r['profile_id']}: several launches")
        if "[STS*]" in r["metric_name"]:
            v = sum(int(float(x["count"])) for x in rows if x["count_kind"] == "dynamic_warp_inst_executed_with_modifier"
                    and (x["instruction_name"] == "STS" or x["instruction_name"].startswith("STS.")))
        else:
            v = sum(int(float(x["count"])) for x in rows if x["count_kind"] == "dynamic_warp_inst_executed_family_total" and x["instruction_family"] == "wgmma")
        if v != int(float(r["value"])) or r["measurement_kind"] != "dynamic_sass_count":
            bad.append(f"{i}: {v} vs {r['value']}")
    fp8 = [ev[i] for i in a5_axes["B_gh200_sts_per_wgmma"] if ev[i]["case"].endswith("fp8_e4m3fn") and ev[i]["dsl"] == "cutile"]
    ratio = float(next(r["value"] for r in fp8 if "[STS*]" in r["metric_name"])) / float(next(r["value"] for r in fp8 if "wgmma" in r["metric_name"]))
    check("17.sts_per_wgmma_same_launch_dynamic_warp_level", not bad, "; ".join(bad[:4]) or f"GH200 FP8 cuTile = {ratio:.2f} STS per WGMMA")

    # 18. histogramming algorithm caveat (TileLang shared-memory privatization) in Figure 4 and its caption
    caps = (PLOTS / "figure_captions.md").read_text() if (PLOTS / "figure_captions.md").exists() else ""
    t4 = svg_texts(PLOTS / "main" / "fig_rq3_within_device_dsl.svg")
    ok = ("histogramming" in M["fig_rq3_within_device_dsl"].get("algorithm_differences", {}) and "histogramming†" in t4
          and re.search(r"histogramming[^.]*shared memory", caps, re.S) is not None)
    check("18.histogramming_algorithm_caveat", ok, "marker + manifest + caption")

    # 19. missing counters shown as missing (n/c), never as zero
    axis_rows = [ev[i] for ids in a5_axes.values() for i in ids]
    missing = [r for r in axis_rows if r["value"] in ("", None)]
    zeros = [r["evidence_id"] for r in axis_rows if r["value"] not in ("", None) and float(r["value"]) == 0.0]
    nc = sum(1 for x in t5 if x == "n/c")
    check("19.missing_counters_shown_as_missing_not_zero", nc >= len(missing) > 0,
          f"{len(missing)} missing evidence rows, {nc} 'n/c' labels; measured zeros (genuine): {len(zeros)}")

    # 20. captions and LaTeX snippets reference every figure; caption numbers match the data
    bad = []
    tex = (PLOTS / "latex" / "figures.tex").read_text() if (PLOTS / "latex" / "figures.tex").exists() else ""
    for n, (sub, _, _) in FIGURES.items():
        if f"`{sub}/{n}`" not in caps:
            bad.append(f"caption missing for {n}")
        if f"{sub}/{n}.pdf" not in tex:
            bad.append(f"LaTeX snippet missing for {n}")
    for c in M["fig_rq1_cross_accelerator"]["plotted_values"]:
        if c["row"] == "Overall" and f"{c['speedup']:.2f}" not in caps:
            bad.append(f"RQ1 overall {c['device']}/{c['dsl']} not in caption")
    for dev, (cnt, near) in W_.items():
        if f"{dev}: Triton {cnt['triton']}, TileLang {cnt['tilelang']}, cuTile {cnt['cutile']}" not in caps:
            bad.append(f"RQ3 {dev} counts not in caption")
        if f"{dev}: {near['triton']}, {near['tilelang']} and {near['cutile']}" not in caps:
            bad.append(f"RQ3 {dev} within-5% counts not in caption")
    check("20.captions_and_latex_reference_every_figure_numbers_match", not bad, "; ".join(bad[:6]))

    # 21. scope: no NKI, no RQ4 numbers, no unexpected figure files; top manifest consistent
    bad = []
    for n, (sub, _, _) in FIGURES.items():
        txt = " ".join(svg_texts(PLOTS / sub / f"{n}.svg"))
        if re.search(r"\bNKI\b|Trainium|trn2", txt) or re.search(r'"(nki|trn2)"', json.dumps(M[n]["plotted_values"])):
            bad.append(n)
    extra = sorted(p.name for p in PLOTS.rglob("*") if p.is_file() and p.suffix in (".pdf", ".svg", ".png") and p.stem not in FIGURES)
    top, ok = PLOTS / "plot_manifest.json", True
    if top.exists():
        tm = json.load(open(top))
        todo = [f for f in tm["figures"] if f.get("status") == "todo"]
        ok = {f["id"] for f in todo} == {"fig5_rq4_llm", "appendix_llm"} and all("outputs" not in f for f in todo)
        for f in tm["figures"]:
            if f.get("status") == "generated":
                ok &= all(sha(PLOTS / o["path"]) == o["sha256"] for o in f["outputs"].values())
    check("21.no_nki_no_rq4_numbers_no_extra_figures_top_manifest", not bad and not extra and ok,
          f"NKI in {bad}; unexpected files {extra}; top manifest {'ok' if ok else 'inconsistent'}")

    # 22. reproducibility: two independent rebuilds into temp dirs, byte-identical to each other and to the committed outputs
    if a.skip_reproducibility:
        results.append({"check": "22.two_rebuilds_byte_identical", "status": "skipped", "detail": "--skip-reproducibility"})
        print("[SKIP] 22.two_rebuilds_byte_identical")
    else:
        diff = []
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            build_into(t1)
            build_into(t2)
            for f in ("figure_evidence.csv", "execution_path_matrix.csv", "rq2_case_selection.json"):
                if not (sha(f"{t1}/combined/{f}") == sha(f"{t2}/combined/{f}") == sha(COMBINED / f)):
                    diff.append(f)
            rels = [f"{sub}/{n}.{e}" for n, (sub, _, _) in FIGURES.items() for e in ("pdf", "svg")] + [f"previews/{n}.png" for n in FIGURES]
            rels.append("tables/fig_a3_secondary_flags.csv")
            for rel in rels:
                if not (sha(f"{t1}/plots/{rel}") == sha(f"{t2}/plots/{rel}") == sha(PLOTS / rel)):
                    diff.append(rel)
            for n in FIGURES:
                x1, x2 = json.load(open(f"{t1}/plots/manifests/{n}.json")), json.load(open(f"{t2}/plots/manifests/{n}.json"))
                if not (x1["plotted_values"] == x2["plotted_values"] == M[n]["plotted_values"] and x1["layout"] == x2["layout"] == M[n]["layout"]):
                    diff.append(f"{n}.json")
        check("22.two_rebuilds_byte_identical", not diff, f"differs: {diff}" if diff else
              f"2 rebuilds: 3 evidence files + {len(rels)} figure/table files + plotted values and layout identical")
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
