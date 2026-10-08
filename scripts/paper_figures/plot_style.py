"""Shared Matplotlib style for the TileArena paper figures (ACL/NAACL two-column)."""
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SINGLE_COL_IN = 3.35
DOUBLE_COL_IN = 7.0

DSL_COLORS = {
    "triton": "#416B87",    # muted steel blue
    "cutile": "#C58458",    # warm copper
    "tilelang": "#4D8C78",  # muted jade green
    "nki": "#9276A5",       # muted violet
    "pytorch": "#8E959B",   # neutral gray
}
DSL_LABEL = {"triton": "Triton", "cutile": "cuTile", "tilelang": "TileLang", "pytorch": "PyTorch"}
DSL_SHORT = {"triton": "T", "cutile": "C", "tilelang": "TL"}
DEVICE_MARKERS = {"B200": "o", "GH200": "s", "MI300X": "^"}
DEVICES = ("B200", "GH200", "MI300X")
NEGATIVE, NEUTRAL, POSITIVE = "#B86F60", "#F7F5F0", "#397E8A"
INK, MUTED, GRID, NA_FILL = "#2B2B2B", "#6E747A", "#D9D9D6", "#EDEDEA"
CATEGORY_ORDER = ["Point-wise", "Reduction/Normalization", "Matrix Multiplication/Attention", "Stencil/Convolution", "Data Layout"]
CATEGORY_SHORT = {"Point-wise": "Point-wise", "Reduction/Normalization": "Reduction/Norm.",
                  "Matrix Multiplication/Attention": "MatMul/Attention", "Stencil/Convolution": "Stencil/Conv.",
                  "Data Layout": "Data Layout"}
CATEGORY_COLORS = {"Point-wise": "#5B7F95", "Reduction/Normalization": "#B07D62", "Matrix Multiplication/Attention": "#6A9A7B",
                   "Stencil/Convolution": "#8C7AA0", "Data Layout": "#A39B5B"}
CATEGORY_MARKERS = {"Point-wise": "o", "Reduction/Normalization": "s", "Matrix Multiplication/Attention": "D",
                    "Stencil/Convolution": "^", "Data Layout": "v"}

DIVERGING = LinearSegmentedColormap.from_list("tilearena_div", [NEGATIVE, NEUTRAL, POSITIVE], N=256)
SEQUENTIAL = LinearSegmentedColormap.from_list("tilearena_seq", ["#F7F5F0", "#E7C9B5", "#C98E70", "#9C5B47"], N=256)

BASE_FONT_PT = 7.0
MIN_FONT_PT = 5.5   # validate_plots fails below this


def _font_family():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"):
        if cand in names:
            return cand
    return "sans-serif"


def apply():
    fam = _font_family()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [fam, "DejaVu Sans"], "font.size": BASE_FONT_PT,
        "axes.titlesize": BASE_FONT_PT + 0.5, "axes.labelsize": BASE_FONT_PT, "xtick.labelsize": BASE_FONT_PT - 0.5,
        "ytick.labelsize": BASE_FONT_PT - 0.5, "legend.fontsize": BASE_FONT_PT - 0.5, "legend.frameon": False,
        "axes.edgecolor": "#9A9FA4", "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": "#5F656B", "ytick.color": "#5F656B", "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.4,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "tilearena",
        "path.simplify": True, "lines.linewidth": 1.0, "patch.linewidth": 0.5,
    })
    return fam


def log2_norm(vmax_abs_log2):
    """Diverging norm centred at log2 = 0 (speedup 1x)."""
    return TwoSlopeNorm(vmin=-vmax_abs_log2, vcenter=0.0, vmax=vmax_abs_log2)


def collect_font_sizes(fig):
    sizes = []
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip():
            sizes.append(t.get_fontsize())
    return sizes


def save(fig, out_root, sub, name):
    """Write PDF (vector, TrueType embedded), SVG (text kept as text) and a 300-dpi PNG preview at the exact design size
    (no tight bbox, so 1 pt in the figure is 1 pt on the page); deterministic metadata. Records how far any drawn
    content reaches outside the canvas (validate_plots requires 0)."""
    out_root = Path(out_root)
    fig.canvas.draw()
    w, h = (float(v) for v in fig.get_size_inches())
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    overflow = {"left": max(0.0, -bb.x0), "bottom": max(0.0, -bb.y0), "right": max(0.0, bb.x1 - w), "top": max(0.0, bb.y1 - h)}
    paths = {}
    for ext in ("pdf", "svg"):
        p = out_root / sub / f"{name}.{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = {"CreationDate": None, "ModDate": None, "Producer": None} if ext == "pdf" else {"Date": None}
        fig.savefig(p, format=ext, metadata=meta)
        paths[ext] = str(p)
    p = out_root / "previews" / f"{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, format="png", dpi=300, metadata={"Software": None})
    paths["png"] = str(p)
    sizes = collect_font_sizes(fig)
    return paths, {"width_in": round(w, 3), "height_in": round(h, 3),
                   "min_font_pt": min(sizes) if sizes else None, "max_font_pt": max(sizes) if sizes else None,
                   "n_text_objects": len(sizes), "content_overflow_in": {k: round(v, 3) for k, v in overflow.items()}}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_manifest(out_root, name, manifest):
    """Output paths are stored relative to the plots root; the plotting code is pinned by sha256 (the generating commit
    may not yet contain it)."""
    for v in manifest.get("outputs", {}).values():
        v["path"] = os.path.relpath(v["path"], out_root)
    here = Path(__file__).resolve().parent
    code = {f"scripts/paper_figures/{f}": sha256(here / f) for f in ("plot_style.py", "figure_data.py")}
    if "script" in manifest:
        code[manifest["script"]] = sha256(here.parents[1] / manifest["script"])
    manifest["code_sha256"] = code
    p = Path(out_root) / "manifests" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(p, "w"), indent=1, sort_keys=False, default=str)
    return str(p)
