# TileArena paper figures (drafts)

Publication-quality Matplotlib drafts for RQ1–RQ3 and the evaluation appendix, generated on CPU from the harmonized
cross-device data in `../combined/`. No GPU benchmark, autotuning, NCU or ROCm profiling was run to produce them.

## Regenerate

From the repository root (CPU only; about 35 s):

```bash
source /projects/kzhou6/bcui2/research/tilebench/tilebench_env.sh   # any env with matplotlib >= 3.8 and numpy
CUDA_VISIBLE_DEVICES= PYTHONPATH=.:scripts/paper_figures python scripts/paper_figures/build_all_figures.py
```

`build_all_figures.py` runs these steps in order:
1. `build_figure_evidence.py` writes `combined/figure_evidence.csv`, `execution_path_matrix.csv` and
   `rq2_case_selection.json` from the NVIDIA/AMD packages.
2. `plot_rq1.py`, `plot_rq2.py`, `plot_rq3.py` and `plot_appendix.py [a1 a2 a3 a4 a5]` draw the figures.
3. It writes `plot_manifest.json`.
4. `validate_plots.py` runs the checks and writes `qa_plots.json`; it exits 1 on failure.

Each script can also run on its own with the same `PYTHONPATH`. The combined layer itself is rebuilt and checked by
`build_combined.py` / `validate_combined.py` (see `../combined/README.md`).

## Layout

| path | content |
|---|---|
| `main/*.{pdf,svg}` | Figures 2–4 (vector; TrueType fonts embedded in PDF, text kept as text in SVG) |
| `appendix/*.{pdf,svg}` | Figures A1–A5 |
| `previews/*.png` | 300-dpi previews of every figure |
| `manifests/<figure>.json` | per-figure manifest (see below) |
| `plot_manifest.json` | index of all figures, including the RQ4 TODO entries |
| `qa_plots.json` | result of `validate_plots.py` |
| `figure_captions.md` | ACL-style caption drafts |
| `limitations.md` | protocol, attribution and coverage limitations |
| `latex/figures.tex` | optional `\includegraphics` snippets (not part of the manuscript) |

| slot | file | size (in) | data |
|---|---|---|---|
| Figure 2 (RQ1) | `main/fig_rq1_cross_accelerator` | 7.0 × 2.3 | formal autotuned latency, category GM of per-operator GM speedups |
| Figure 3 (RQ2) | `main/fig_rq2_cross_device_diagnosis` | 7.0 × 4.1 | formal latency at the profiled `case_id_v2` plus device-native evidence |
| Figure 4 (RQ3) | `main/fig_rq3_within_device_dsl` | 7.0 × 3.75 | within-device latency ratios over the three-DSL case intersection |
| A1 | `appendix/fig_a1_performance_atlas` | 7.0 × 8.6 | per-operator speedups; cross-device Δ over matched cases |
| A2 | `appendix/fig_a2_shape_dtype` | 7.0 × 6.3 | per-case speedups for shape- or dtype-sensitive operators |
| A3 | `appendix/fig_a3_execution_paths` | 7.0 × 7.6 | categorical execution paths (dynamic or static SASS/ISA) |
| A4 | `appendix/fig_a4_within_device_matrix` | 3.35 × 8.4 | slowdown vs. the fastest DSL on the device |
| A5 | `appendix/fig_a5_profiling_evidence` | 7.0 × 6.9 | counters and diagnostic experiments per mechanism |
| Figure 5 (RQ4) | — | — | **TODO**: blocked, no finalized LLM results |
| Appendix LLM figure | — | — | **TODO**: blocked, no finalized LLM results |

Every manifest records:
- the script and the sha256 of the plotting code, and the generating commit;
- the input-file hashes;
- the metric formula and aggregation order;
- the case coverage and excluded cases;
- the selection criteria and the profiling evidence IDs;
- the device and figure limitations;
- every plotted value;
- the output hashes and the layout (size, min/max font, content outside the canvas).

## Data protocol

Defined in `../combined/comparison_manifest.json`:
- `S[o,b,d]` = GM over valid autotuned cases of `torch_ms / dsl_ms`.
- Category and overall values are a GM over operators of `S[o,b,d]`.
- Within-device ratios use only the cases valid for every DSL being compared.
- Cross-device Δ = `log2(S_dev2 / S_dev1)`, with both S computed over the matched `case_id_v2`.
- Winner = the DSL with the lowest GM latency over the three-DSL intersection; "within 5%" means runner-up / winner ≤ 1.05.

Formal latencies come only from `benchmark_cases_normalized.csv.gz`. Profiler durations are never used as latency.
MI300X diagnostic latencies (warmup 2 / repeat 10) appear only in A5, on axes labelled "diagnostic run".

## QA (`validate_plots.py`)

`validate_plots.py` recomputes every plotted number from `benchmark_cases_normalized.csv.gz` and
`figure_evidence.csv`, without using the `figure_data.py` helpers, and compares it with the manifests. It also
compares it with the text rendered into the SVG files. The checks are:

1. outputs exist and their hashes match the manifests;
2. the PDF is 3.35 or 7.0 in wide, the PNG is ≥ 300 dpi, and nothing lies outside the canvas;
3. every rendered font is ≥ 5.5 pt and PDF fonts are TrueType;
4. all plotted values are finite;
5. the source data is unchanged: combined inputs, device packages and results CSVs;
6. no profiler or diagnostic latency is used as a formal latency;
7. RQ1 values, cell labels and N/A cells are correct (N/A is never drawn as zero);
8. RQ3 ratios over the explicit three-DSL intersection, the winners and the winner text are correct;
9. A1 speedups, matched-case Δ and labels are correct;
10. A2 per-case values and A4 slowdowns are correct;
11. RQ2 uses one profiled `case_id_v2` per case on every device;
12. evidence is traceable, no axis mixes vendors, and no axis mixes static with dynamic counts;
13. missing evidence is drawn as "n/c", not zero;
14. A3 cells equal `execution_path_matrix.csv`;
15. there is no NKI, no RQ4 number and no extra figure, and the top manifest is consistent;
16. regenerating everything into a temp dir is byte-identical (PDF, SVG, PNG and the evidence files).

## Style

All figures share `scripts/paper_figures/plot_style.py`:
- **Font and sizes:** Liberation Sans (Helvetica metric-compatible), 7 pt base, 5.5 pt minimum. Designed at 7.0 in
  (double column) or 3.35 in (single column).
- **DSL colours:** Triton steel blue, cuTile copper, TileLang jade, PyTorch gray.
- **Device markers:** B200 ●, GH200 ■, MI300X ▲.
- **Ratio colour scale:** diverging and centred at 1×, so red is slower and teal is faster.
- **Not available:** shown as hatched N/A and never as zero.

**Page width:** the ACL template has a 6.30 in text width and a 3.03 in column width. Included at `\textwidth` or
`\columnwidth`, the figures are scaled by about 0.90, so the smallest text prints at about 4.95 pt (see
`limitations.md`).
