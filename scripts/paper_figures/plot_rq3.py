"""Figure 4 (RQ3): within-device DSL comparison on B200 and GH200 (one point per operator)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_data as FD  # noqa: E402
import plot_style as PS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAME = "fig_rq3_within_device_dsl"
DEVS = ("B200", "GH200")
NEAR = 1.05
LABEL_ALIASES = {"matmul_fp32_fp16_fp8": "matmul", "block_sparse_attention": "block_sparse_attn",
                 "linear_self_attention": "linear_self_attn", "moe_topk_gating": "moe_topk_gating"}


def compute(D):
    pts, winners = {}, {}
    for dev in DEVS:
        cnt, near = {"triton": 0, "cutile": 0, "tilelang": 0}, {"triton": 0, "cutile": 0, "tilelang": 0}
        for op in D.operators():
            ids = D.matched_ids(op, [(dev, "triton"), (dev, "cutile"), (dev, "tilelang")])
            lat = {s: D.latency_gm(dev, s, op, ids) for s in ("triton", "cutile", "tilelang")}
            x = lat["cutile"] / lat["triton"]
            y = lat["tilelang"] / lat["triton"]
            w = min(lat, key=lat.get)
            ru = sorted(lat.values())[1] / lat[w]
            cnt[w] += 1
            near[w] += ru <= NEAR
            pts[(dev, op)] = {"x_cutile_over_triton": x, "y_tilelang_over_triton": y, "n_cases": len(ids), "winner": w,
                              "runner_up_ratio": ru, "category": D.categories[op]}
        winners[dev] = {"counts": cnt, "winner_within_5pct": near}
    return pts, winners


def plot(D, out_root):
    PS.apply()
    pts, winners = compute(D)
    allv = [v for p in pts.values() for v in (p["x_cutile_over_triton"], p["y_tilelang_over_triton"])]
    lo = 2 ** math.floor(math.log2(min(allv)) - 0.15)
    hi = 2 ** math.ceil(math.log2(max(allv)) + 0.15)
    fig, axes = plt.subplots(1, 2, figsize=(PS.DOUBLE_COL_IN, 3.75), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.17, top=0.95, wspace=0.10)
    labelled = {}
    for k, (ax, dev) in enumerate(zip(axes, DEVS)):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.axvline(1, color="#9A9FA4", lw=0.6, zorder=0)
        ax.axhline(1, color="#9A9FA4", lw=0.6, zorder=0)
        ax.grid(True, which="major", color=PS.GRID, lw=0.35, zorder=0)
        ticks = [t for t in (1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16) if lo <= t <= hi]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        tl = [f"{t:g}×" if t >= 1 else f"1/{int(1/t)}×" for t in ticks]
        ax.set_xticklabels(tl)
        ax.set_yticklabels(tl)
        ax.minorticks_off()
        for op in D.operators():
            p = pts[(dev, op)]
            c = p["category"]
            ax.scatter(p["x_cutile_over_triton"], p["y_tilelang_over_triton"], marker=PS.CATEGORY_MARKERS[c], s=17,
                       color=PS.CATEGORY_COLORS[c], edgecolor="white", linewidth=0.4, zorder=3, alpha=0.95)
        # label the most extreme operators (distance from the 1x/1x origin in log space); greedy placement that
        # avoids the markers, the quadrant notes and labels already placed (all in log2 axis units)
        cw, ch = 0.075, 0.17                      # approx. character width / line height in log2 units at 5.8 pt
        obstacles = [(math.log2(pts[(dev, o)]["x_cutile_over_triton"]), math.log2(pts[(dev, o)]["y_tilelang_over_triton"]))
                     for o in D.operators()]
        boxes = [(math.log2(lo) + 0.1, math.log2(lo) + 0.1, math.log2(lo) + 1.25, math.log2(lo) + 0.55),
                 (math.log2(hi) - 1.25, math.log2(hi) - 0.55, math.log2(hi) - 0.1, math.log2(hi) - 0.1),
                 (math.log2(lo) + 0.05, math.log2(hi) - 1.05, math.log2(lo) + 2.6, math.log2(hi) - 0.05)]
        cand = sorted(D.operators(), key=lambda o: -math.hypot(math.log2(pts[(dev, o)]["x_cutile_over_triton"]),
                                                              math.log2(pts[(dev, o)]["y_tilelang_over_triton"])))[:7]
        for op in cand:
            p = pts[(dev, op)]
            x, y = p["x_cutile_over_triton"], p["y_tilelang_over_triton"]
            lx, ly = math.log2(x), math.log2(y)
            lab = LABEL_ALIASES.get(op, op)
            wl = cw * len(lab)
            best = None
            for ang in range(0, 360, 30):
                for r in (0.28, 0.45, 0.65):
                    tx, ty = lx + r * math.cos(math.radians(ang)), ly + r * math.sin(math.radians(ang))
                    left = ang <= 90 or ang >= 270
                    bx0, bx1 = (tx, tx + wl) if left else (tx - wl, tx)
                    by0, by1 = ty - ch / 2, ty + ch / 2
                    if bx0 < math.log2(lo) + 0.05 or bx1 > math.log2(hi) - 0.05 or by0 < math.log2(lo) or by1 > math.log2(hi):
                        continue
                    cost = r
                    cost += sum(3.0 for (ox, oy) in obstacles if bx0 - 0.1 < ox < bx1 + 0.1 and by0 - 0.1 < oy < by1 + 0.1)
                    cost += sum(5.0 for (a0, b0, a1, b1) in boxes if not (bx1 < a0 or bx0 > a1 or by1 < b0 or by0 > b1))
                    if best is None or cost < best[0]:
                        best = (cost, tx, ty, left, (bx0, by0, bx1, by1))
            _, tx, ty, left, box = best
            boxes.append(box)
            ax.annotate(lab, (x, y), xytext=(2 ** tx, 2 ** ty), fontsize=5.8, color="#3B3F43", ha="left" if left else "right",
                        va="center", arrowprops=dict(arrowstyle="-", lw=0.35, color="#8E959B", shrinkA=0, shrinkB=2.5))
            labelled.setdefault(dev, []).append(op)
        ax.text(lo * 1.12, lo * 1.12, "both faster\nthan Triton", fontsize=5.8, color=PS.MUTED, ha="left", va="bottom")
        ax.text(hi / 1.12, hi / 1.12, "both slower\nthan Triton", fontsize=5.8, color=PS.MUTED, ha="right", va="top")
        w = winners[dev]
        ax.set_title(f"{'AB'[k]}  {dev}", loc="left", fontsize=7.5, fontweight="bold")
        txt = (f"fastest DSL (no. of operators)\nTriton {w['counts']['triton']} · TileLang {w['counts']['tilelang']} · "
               f"cuTile {w['counts']['cutile']}\nwithin 5% of runner-up:\nTriton {w['winner_within_5pct']['triton']} · "
               f"TileLang {w['winner_within_5pct']['tilelang']} · cuTile {w['winner_within_5pct']['cutile']}")
        ax.text(0.02, 0.975, txt, transform=ax.transAxes, fontsize=5.8, color=PS.INK, va="top", ha="left", linespacing=1.2,
                bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor=PS.GRID, linewidth=0.4))
        ax.set_xlabel("cuTile / Triton latency (GM over cases)")
        if k == 0:
            ax.set_ylabel("TileLang / Triton latency (GM over cases)")
    handles = [Line2D([], [], marker=PS.CATEGORY_MARKERS[c], ls="", color=PS.CATEGORY_COLORS[c], markersize=4.2, label=PS.CATEGORY_SHORT[c])
               for c in D.cat_order]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.2, handletextpad=0.2, columnspacing=1.0, bbox_to_anchor=(0.53, 0.0))
    paths, layout = PS.save(fig, out_root, "main", NAME)
    plt.close(fig)
    return paths, layout, pts, winners, labelled, (lo, hi)


def main(out_root=FD.PLOTS):
    D = FD.Data("autotune")
    paths, layout, pts, winners, labelled, lim = plot(D, out_root)
    manifest = {
        "figure": NAME, "rq": "RQ3", "script": "scripts/paper_figures/plot_rq3.py", "source_git_commit": FD.git_head(),
        "source_data_files": FD.input_hashes(["benchmark_cases_normalized.csv.gz", "category_mapping.csv", "comparison_manifest.json"]),
        "metric_formula": "x = GM_cases(cutile_ms/triton_ms), y = GM_cases(tilelang_ms/triton_ms) over cases valid for all three DSLs on the device; >1 = slower than Triton",
        "aggregation_order": ["geometric mean over matched input cases within an operator"],
        "winner_definition": "per operator, the DSL with the lowest GM latency over the matched cases; 'within 5%' = runner-up / winner <= 1.05 (no uncertainty analysis; near parity is not significant)",
        "axes": {"x": "cuTile/Triton latency ratio (log2)", "y": "TileLang/Triton latency ratio (log2)", "limits": lim, "shared_between_panels": True},
        "case_coverage": {dev: {op: pts[(dev, op)]["n_cases"] for op in D.operators()} for dev in DEVS},
        "excluded_cases": {"invalid_or_missing_rows": len(D.excluded)}, "selection_criteria": "all 45 operators; labels on the 7 operators farthest from (1x, 1x) per panel",
        "labelled_operators": labelled, "winners": winners, "profiling_evidence_ids": [],
        "known_limitations": D.manifest["device_limitations"] | {"figure": ["B200 TileLang measured in a later campaign than B200 Triton/cuTile",
                                                                           "different benchmark protocols on B200 and GH200; only within-device ratios are plotted"]},
        "plotted_values": [{"device": d, "operator": o, **pts[(d, o)]} for d in DEVS for o in D.operators()],
        "outputs": {k: {"path": v, "sha256": PS.sha256(v)} for k, v in paths.items()}, "layout": layout,
    }
    PS.write_manifest(out_root, NAME, manifest)
    return manifest


if __name__ == "__main__":
    main()
