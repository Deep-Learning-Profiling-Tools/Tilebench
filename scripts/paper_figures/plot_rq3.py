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


ALGO_DIFF = {"histogramming": "TileLang privatizes the histogram in shared memory; Triton and cuTile update global partial rows atomically"}
LABEL_MIN_LOG2 = 1.0              # always label operators at least 2x slower or faster than Triton on either axis
LABEL_OPT_LOG2 = math.log2(1.5)   # label those at least 1.5x only where a free slot exists next to the marker
OPT_MAX_R = 0.36                  # free slot: within this distance (inches), no collision with labels, leaders or markers
FS, FS_LAB, FS_TITLE = 8.5, 8.0, 10.0          # tick/axis/legend text, operator labels, panel titles (pt)
H_IN = 3.95
L_IN, R_IN, GAP_IN, T_IN, B_IN = 0.62, 0.08, 0.2, 0.5, 0.98     # margins around the two panels (inches)
MARGIN_LOG2 = 0.3                              # axis padding beyond the data (octaves)
PAD, MARK_R = 0.015, 0.05                      # label box padding, marker radius (inches)


def boxes_hit(a, b):
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def seg_hits_box(seg, b):
    """Liang-Barsky: does the segment intersect the box (x0, y0, x1, y1)?"""
    (x0, y0), (x1, y1) = seg
    t0, t1, dx, dy = 0.0, 1.0, x1 - x0, y1 - y0
    for p_, q_ in ((-dx, x0 - b[0]), (dx, b[2] - x0), (-dy, y0 - b[1]), (dy, b[3] - y0)):
        if p_ == 0:
            if q_ < 0:
                return False
            continue
        t = q_ / p_
        if p_ < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def seg_point_dist(seg, q):
    (x0, y0), (x1, y1) = seg
    dx, dy = x1 - x0, y1 - y0
    t = max(0.0, min(1.0, ((q[0] - x0) * dx + (q[1] - y0) * dy) / (dx * dx + dy * dy or 1.0)))
    return math.hypot(x0 + t * dx - q[0], y0 + t * dy - q[1])


def segs_cross(s1, s2):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    (a, b), (c, d) = s1, s2
    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def plot(D, out_root):
    PS.apply()
    pts, winners = compute(D)
    L = math.log2
    xs = [p["x_cutile_over_triton"] for p in pts.values()]
    ys = [p["y_tilelang_over_triton"] for p in pts.values()]
    # limits follow the data (shared by both panels); x reaches down to 1/2 so the lower-left quadrant note fits
    xlim = (min(0.5, 2 ** (L(min(xs)) - MARGIN_LOG2)), 2 ** (L(max(xs)) + MARGIN_LOG2))
    ylim = (2 ** (L(min(ys)) - MARGIN_LOG2), 2 ** (L(max(ys)) + MARGIN_LOG2))
    W = PS.DOUBLE_COL_IN
    AW, AH = (W - L_IN - R_IN - GAP_IN) / 2, H_IN - T_IN - B_IN
    fig = plt.figure(figsize=(W, H_IN))
    renderer = fig.canvas.get_renderer()

    def text_wh(s):               # rendered label size (inches)
        t = fig.text(0, 0, s, fontsize=FS_LAB)
        bb = t.get_window_extent(renderer)
        t.remove()
        return bb.width / fig.dpi, bb.height / fig.dpi

    def to_in(x, y):              # data -> inches inside the axes
        return ((L(x) - L(xlim[0])) / (L(xlim[1]) - L(xlim[0])) * AW, (L(y) - L(ylim[0])) / (L(ylim[1]) - L(ylim[0])) * AH)

    def from_in(px, py):
        return (2 ** (L(xlim[0]) + px / AW * (L(xlim[1]) - L(xlim[0]))), 2 ** (L(ylim[0]) + py / AH * (L(ylim[1]) - L(ylim[0]))))

    def ticks(lo, hi):
        return [t for t in (1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16) if lo <= t <= hi]

    def tick_label(t):
        return f"{t:g}×" if t >= 1 else f"1/{int(1 / t)}×"

    labelled = {}
    for k, dev in enumerate(DEVS):
        ax = fig.add_axes([(L_IN + k * (AW + GAP_IN)) / W, B_IN / H_IN, AW / W, AH / H_IN])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axvline(1, color="#9A9FA4", lw=0.6, zorder=0)
        ax.axhline(1, color="#9A9FA4", lw=0.6, zorder=0)
        ax.grid(True, which="major", color=PS.GRID, lw=0.35, zorder=0)
        ax.set_xticks(ticks(*xlim))
        ax.set_xticklabels([tick_label(t) for t in ticks(*xlim)], fontsize=FS)
        ax.set_yticks(ticks(*ylim))
        ax.set_yticklabels([tick_label(t) for t in ticks(*ylim)] if k == 0 else [], fontsize=FS)
        ax.minorticks_off()
        for op in D.operators():
            p = pts[(dev, op)]
            c = p["category"]
            ax.scatter(p["x_cutile_over_triton"], p["y_tilelang_over_triton"], marker=PS.CATEGORY_MARKERS[c], s=34,
                       color=PS.CATEGORY_COLORS[c], edgecolor="white", linewidth=0.5, zorder=3, alpha=0.95)
            if op in ALGO_DIFF:
                ax.scatter(p["x_cutile_over_triton"], p["y_tilelang_over_triton"], marker="o", s=150, facecolor="none",
                           edgecolor=PS.INK, linewidth=0.7, zorder=4)
        # greedy label placement in inches inside the axes, by cost: label on label or on the quadrant note >> leader line
        # through a label > crossing leader lines > label over a marker > distance
        marks = [to_in(pts[(dev, o)]["x_cutile_over_triton"], pts[(dev, o)]["y_tilelang_over_triton"]) for o in D.operators()]
        boxes = [(0.0, 0.0, 0.95, 0.36)]                                   # quadrant note (lower left)
        lines = []
        dist = {o: max(abs(L(pts[(dev, o)]["x_cutile_over_triton"])), abs(L(pts[(dev, o)]["y_tilelang_over_triton"]))) for o in D.operators()}
        far = lambda o: -math.hypot(L(pts[(dev, o)]["x_cutile_over_triton"]), L(pts[(dev, o)]["y_tilelang_over_triton"]))  # noqa: E731
        cand = sorted((o for o in D.operators() if dist[o] >= LABEL_MIN_LOG2), key=far) + \
            sorted((o for o in D.operators() if LABEL_OPT_LOG2 <= dist[o] < LABEL_MIN_LOG2), key=far)
        for op in cand:
            p = pts[(dev, op)]
            x, y = p["x_cutile_over_triton"], p["y_tilelang_over_triton"]
            px, py = to_in(x, y)
            lab = LABEL_ALIASES.get(op, op) + ("†" if op in ALGO_DIFF else "")
            wl, hl = text_wh(lab)
            best = None
            for ang in range(0, 360, 15):
                for r in (0.12, 0.18, 0.26, 0.36, 0.5, 0.65, 0.8):
                    tx, ty = px + r * math.cos(math.radians(ang)), py + r * math.sin(math.radians(ang))
                    left = ang <= 90 or ang >= 270
                    box = ((tx, ty - hl / 2 - PAD, tx + wl, ty + hl / 2 + PAD) if left else (tx - wl, ty - hl / 2 - PAD, tx, ty + hl / 2 + PAD))
                    if box[0] < 0.04 or box[2] > AW - 0.04 or box[1] < 0.02 or box[3] > AH - 0.02:
                        continue
                    seg = ((px, py), (tx, ty))
                    cost = r + 200.0 * sum(boxes_hit(box, b_) for b_ in boxes)
                    cost += 60.0 * (sum(seg_hits_box(seg, b_) for b_ in boxes) + sum(seg_hits_box(l_, box) for l_ in lines))
                    cost += 30.0 * sum(segs_cross(seg, l_) for l_ in lines)
                    cost += 6.0 * sum(box[0] - MARK_R < ox < box[2] + MARK_R and box[1] - MARK_R < oy < box[3] + MARK_R for ox, oy in marks)
                    cost += 6.0 * sum(seg_point_dist(seg, m_) < 1.5 * MARK_R for m_ in marks if m_ != (px, py))  # leader past a marker
                    if best is None or cost < best[0]:
                        best = (cost, tx, ty, left, box, seg)
            cost, tx, ty, left, box, seg = best
            if dist[op] < LABEL_MIN_LOG2 and cost > OPT_MAX_R:
                continue                                                   # optional label: no free slot next to the marker
            boxes.append(box)
            lines.append(seg)
            ax.annotate(lab, (x, y), xytext=from_in(tx, ty), fontsize=FS_LAB, color="#3B3F43", ha="left" if left else "right",
                        va="center", arrowprops=dict(arrowstyle="-", lw=0.45, color="#8E959B", shrinkA=1, shrinkB=3.5,
                                                     relpos=(0.0 if left else 1.0, 0.5)))   # leader ends at the near end of the label
            labelled.setdefault(dev, []).append(op)
        w = winners[dev]["counts"]
        ax.text(0.0, 1.0 + 0.27 / AH, f"{'AB'[k]}  {dev}", transform=ax.transAxes, fontsize=FS_TITLE, fontweight="bold", va="bottom",
                ha="left")
        ax.text(0.0, 1.0 + 0.05 / AH, f"fastest: Triton {w['triton']}, TileLang {w['tilelang']}, cuTile {w['cutile']}",
                transform=ax.transAxes, fontsize=FS, color=PS.INK, va="bottom", ha="left")
        ax.text(0.04 / AW, 0.04 / AH, "both faster\nthan Triton", transform=ax.transAxes, fontsize=FS, color=PS.MUTED, ha="left",
                va="bottom")
        ax.set_xlabel("cuTile / Triton latency", fontsize=FS + 0.5)
        if k == 0:
            ax.set_ylabel("TileLang / Triton latency", fontsize=FS + 0.5)
    handles = [Line2D([], [], marker=PS.CATEGORY_MARKERS[c], ls="", color=PS.CATEGORY_COLORS[c], markersize=6, label=PS.CATEGORY_SHORT[c])
               for c in D.cat_order]
    handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=PS.INK, markersize=9, markeredgewidth=0.7,
                          label="† different algorithm"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=FS, handletextpad=0.3, columnspacing=1.6,
               bbox_to_anchor=((L_IN + (W - L_IN - R_IN) / 2) / W, 0.0))
    paths, layout = PS.save(fig, out_root, "main", NAME)
    plt.close(fig)
    return paths, layout, pts, winners, labelled, {"x": list(xlim), "y": list(ylim)}


def main(out_root=FD.PLOTS):
    D = FD.Data("autotune")
    paths, layout, pts, winners, labelled, lim = plot(D, out_root)
    manifest = {
        "figure": NAME, "rq": "RQ3", "script": "scripts/paper_figures/plot_rq3.py", "source_git_commit": FD.git_head(),
        "source_data_files": FD.input_hashes(["benchmark_cases_normalized.csv.gz", "category_mapping.csv", "comparison_manifest.json"]),
        "metric_formula": "x = GM_cases(cutile_ms/triton_ms), y = GM_cases(tilelang_ms/triton_ms) over cases valid for all three DSLs on the device; >1 = slower than Triton",
        "aggregation_order": ["geometric mean over matched input cases within an operator"],
        "winner_definition": "per operator, the DSL with the lowest GM latency over the matched cases; numerical winner, no uncertainty analysis. 'winner_within_5pct' (runner-up / winner <= 1.05) is recorded here and reported in the caption, not drawn",
        "axes": {"x": "cuTile/Triton latency ratio (log2)", "y": "TileLang/Triton latency ratio (log2)", "limits": lim, "shared_between_panels": True},
        "case_coverage": {dev: {op: pts[(dev, op)]["n_cases"] for op in D.operators()} for dev in DEVS},
        "excluded_cases": {"invalid_or_missing_rows": len(D.excluded)},
        "selection_criteria": f"all 45 operators; labels on every operator at least {2 ** LABEL_MIN_LOG2:g}x from Triton on either axis, and on "
                              f"those at least {2 ** LABEL_OPT_LOG2:g}x where a collision-free slot exists within {OPT_MAX_R} in of the marker",
        "algorithm_differences": ALGO_DIFF,
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
