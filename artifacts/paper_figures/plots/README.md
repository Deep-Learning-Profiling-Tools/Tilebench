# TileArena paper figures

Matplotlib figures for RQ1–RQ3 and the evaluation appendix. They are generated on CPU from the harmonized cross-device
data in `../combined/`. No GPU benchmark, autotuning, NCU or ROCm profiling run was involved in producing them.

## Regenerate

From the repository root (CPU only, about 35 s, plus about 70 s for the two reproducibility rebuilds in the validator):

```bash
source /projects/kzhou6/bcui2/research/tilebench/tilebench_env.sh   # any env with matplotlib >= 3.8 and numpy
CUDA_VISIBLE_DEVICES= PYTHONPATH=.:scripts/paper_figures python scripts/paper_figures/build_all_figures.py
```

`build_all_figures.py` runs four steps in order:

1. `build_figure_evidence.py` writes `combined/figure_evidence.csv`, `combined/execution_path_matrix.csv` and
   `combined/rq2_case_selection.json` from the NVIDIA/AMD packages.
2. `plot_rq1.py`, `plot_rq2.py`, `plot_rq3.py` and `plot_appendix.py [a1 a2 a3 a4 a5]` draw the figures.
3. It writes `plot_manifest.json`.
4. `validate_plots.py` checks the result and writes `qa_plots.json`; it exits 1 on failure.

The combined layer is rebuilt and checked by `build_combined.py` / `validate_combined.py` (see `../combined/README.md`).

## Layout

| path | content |
|---|---|
| `main/*.{pdf,svg}` | Figures 2–4 (vector; TrueType fonts embedded in the PDF, text kept as text in the SVG) |
| `appendix/*.{pdf,svg}` | Figures A1–A5 |
| `previews/*.png` | 300-dpi previews of every figure |
| `manifests/<figure>.json` | per-figure manifest (see below) |
| `tables/fig_a3_secondary_flags.csv` | companion table of Figure A3 (spills, atomics, LDSM/STSM, layout conversions) |
| `plot_manifest.json` | index of all figures, including the RQ4 TODO entries |
| `qa_plots.json` | result of `validate_plots.py` |
| `figure_captions.md` | caption drafts |
| `limitations.md` | protocol, attribution and coverage limitations |
| `scientific_audit.md` | audit of every mechanism stated in Figure 3 and A5 |
| `latex/figures.tex` | optional `\includegraphics` snippets (not part of the manuscript) |

## Figures

Sizes are the printed sizes. All figures are drawn at the ACL text width of 6.30 in and are meant to be included at
`width=\textwidth` (scale 1.0).

| slot | file | size (in) | min. font | content |
|---|---|---|---|---|
| Figure 2 (RQ1) | `main/fig_rq1_cross_accelerator` | 6.30 × 2.20 | 7.0 pt | category GM of per-operator GM speedups, seven device/DSL columns |
| Figure 3 (RQ2) | `main/fig_rq2_cross_device_diagnosis` | 6.30 × 4.95 | 7.0 pt | three mechanisms: formal speedup at the profiled input + two NVIDIA counters + MI300X note |
| Figure 4 (RQ3) | `main/fig_rq3_within_device_dsl` | 6.30 × 3.55 | 7.0 pt | within-device latency ratios over the three-DSL intersection |
| A1 | `appendix/fig_a1_performance_atlas` | 6.30 × 8.70 | 6.5 pt | 45-operator speedups; cross-device Δ over matched cases |
| A2 | `appendix/fig_a2_shape_dtype` | 6.30 × 6.95 | 6.0 pt | per-case speedups for shape- or dtype-sensitive operators |
| A3 | `appendix/fig_a3_execution_paths` | 6.30 × 5.60 | 6.0 pt | matrix instruction family + operand path + staging |
| A4 | `appendix/fig_a4_within_device_matrix` | 6.30 × 7.90 | 7.0 pt | slowdown vs. the fastest DSL on the device, 45 operators |
| A5 | `appendix/fig_a5_profiling_evidence` | 6.30 × 6.10 | 6.0 pt | counters and diagnostic experiments per mechanism |
| Figure 5 (RQ4) | — | — | — | **TODO**: blocked, no finalized LLM results |
| Appendix LLM figure | — | — | — | **TODO**: blocked, no finalized LLM results |

Every manifest records:
- the script and the sha256 of the plotting code;
- the generating commit;
- the input-file hashes;
- the metric formula and aggregation order;
- the case coverage and excluded cases;
- the selection criteria;
- the profiling evidence IDs (per axis for A5);
- confounders and limitations;
- every plotted value;
- the output hashes;
- the layout: size, minimum and maximum font, content outside the canvas, and text overlaps.

## Data protocol

Defined in `../combined/comparison_manifest.json`:
- `S[o,b,d]` is the geometric mean (GM) over valid autotuned cases of `torch_ms / dsl_ms`.
- Category and overall values are a GM over operators.
- Within-device ratios use only the cases valid for every DSL being compared.
- Cross-device Δ = `log2(S_dev2 / S_dev1)`, with both S computed over the matched `case_id_v2`.
- The winner is the DSL with the lowest GM latency over the three-DSL intersection; this is a numerical winner, not a
  significance test.

Formal latencies come only from `benchmark_cases_normalized.csv.gz`, and profiler durations are never used as
latency. Figure 3 uses the one `case_id_v2` that every profile of the case captured. MI300X diagnostic latencies
appear only in A5, on axes titled "(diagnostic)".

## QA (`validate_plots.py`)

`validate_plots.py` recomputes every plotted number from `benchmark_cases_normalized.csv.gz`, `figure_evidence.csv`
and the packages' `instruction_mix.csv` and `diagnostic_experiments.csv`. It does not use the `figure_data.py`
helpers. It compares the recomputed numbers with the manifests and with the text rendered into the SVG files.

**Outputs and layout**

1. The outputs exist and match the manifest hashes.
2. They were produced by the current plotting code.
3. Every figure is 6.30 in wide, the PNGs are ≥ 300 dpi, and no content lies outside the canvas.
4. Fonts are ≥ 7 pt in the main figures and ≥ 6 pt in the appendix figures, and the PDF fonts are TrueType.
5. No two text labels overlap.
6. All plotted values are finite.
7. The extraction packages, results CSVs and inputs are unchanged.

**Values**

8. No profiler duration is used as benchmark latency.
9. RQ1 is a recomputed operator-balanced GM over exactly the 7 supported columns, and its rendered labels match.
10. The per-operator DSL winners and the A4 slowdowns are correct.
11. The A1 deltas use matched `case_id_v2`, and the A2 values are correct.
12. Figure 3 uses the exact profiled case and its formal latency.
13. Counter units and denominators are correct when recomputed.

**Evidence semantics**

14. No numeric axis mixes vendors; static and dynamic counts stay distinct.
15. MI300X diagnostic latency is labelled separately.
16. The cache modifier names match the emitted gfx942 ISA.
17. STS/WGMMA is counted in the same single launch, both dynamic and warp-level, and recomputes to the plotted value.
18. The histogramming algorithm caveat appears in the marker, the manifest and the caption.
19. Missing counters are shown as missing (n/c), never as zero.

**Documentation and scope**

20. The captions and LaTeX snippets reference every figure, and the caption numbers match the data.
21. There is no NKI and no RQ4 number, and the top manifest is consistent.

**Occupancy representation**

22. Occupancy is shown as achieved vs. theoretical limit without reference lines: in Figure 3C and A5 the achieved
    bars over grey theoretical bars and their labels match the evidence, both figures share one occupancy style, and
    issue activity is unchanged.

**Reproducibility**

23. Two independent rebuilds are byte-identical to each other and to the committed outputs.

## Style

All figures share `scripts/paper_figures/plot_style.py`:
- **Font:** Liberation Sans, metric-compatible with Helvetica.
- **DSL colours:** Triton steel blue `#416B87`, cuTile copper `#C58458`, TileLang jade `#4D8C78`, PyTorch gray
  `#8E959B`.
- **Ratio colour scale:** diverging red–neutral–teal and centred at 1×.
- **Slowdown colour scale:** sequential from neutral to copper.
- **Unavailable data:** never drawn as zero.
