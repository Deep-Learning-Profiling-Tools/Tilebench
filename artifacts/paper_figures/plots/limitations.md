# Limitations of the figure drafts

These limitations apply to the figures in this directory. Device-level entries repeat
`combined/comparison_manifest.json` (`device_limitations`); every figure manifest carries them under
`known_limitations`.

## What the figures can and cannot be used for

- Every speedup is relative to the PyTorch baseline on the same device, measured under that device's benchmark
  protocol. Figures 2, 3A and A1 compare how each DSL behaves relative to vendor libraries across accelerators. They
  are not absolute hardware comparisons.
- The cross-device deltas in Figure A1 (right) are computed over matched `case_id_v2`, i.e. the input cases valid on
  both devices. They still mix hardware, compiler and library changes; they do not isolate any one of them.
- Winner counts in Figure 4 and the near-parity threshold in Figures 4 and A4 (5%) carry no uncertainty analysis.
  Each case is a single campaign measurement.
- Figures 3B, A3 and A5 are diagnostic evidence for single profiled cases (the maximum input of each
  operator/dtype). They support mechanism attribution but do not measure the effect on the formal latency.

## Benchmark protocols differ between devices

| device | formal timing | L2/LLC eviction | notes |
|---|---|---|---|
| B200 | warmup 20 / repeat 100 (frozen paper columns) | fixed 64 MB for 41 operators (< 126.5 MB L2) | Triton/cuTile from the paper campaign with cuda-tile 1.3.0 (benchmark source commit not recorded); TileLang measured later (direct runtime, PR #319) in another environment |
| GH200 | warmup 1 / repeat 3 | 120 MiB | CUDA 13.1, cuda-tile 1.5.0 (tileiras 13.4.92), NCU 2025.4.0 |
| MI300X | warmup 20 / repeat 100 | 512 MiB | ROCm eager timing (the HIP-graph request falls back to eager); Triton only |

## Measurement semantics that are kept separate

- **NVIDIA**: Nsight Compute counters and dynamic per-opcode SASS counts (executed instructions).
- **AMD**: static AMDGCN ISA counts (instructions in the binary), `SQ_INSTS_*` hardware counters, PC samples and ATT
  hit counts.

No numeric axis mixes the two vendors, and no axis mixes static with dynamic counts. `validate_plots.py` check 12
enforces both from the per-axis evidence lists in the A5 manifest. MI300X evidence in Figures 3 and A5 is shown as
text, or on its own axis when it is a latency.

- The MI300X latencies in Figure A5 come from diagnostic experiments (warmup 2 / repeat 10) and are labelled as such.
  They never enter a formal-latency figure (check 06).
- The MI300X descriptor-versus-pointer FP16 variant also changes `num_stages` from 3 to 2, so it is not a one-factor
  experiment.
- In Figure A3, dashed cells are static code paths, not executed counts. Absence of a flag does not prove absence.

## Missing or reduced evidence

- **MI300X FP8 matmul.** `matmul_fp32_fp16_fp8` in FP8 E4M3FN is unsupported on MI300X. It is shown as N/A, never as
  zero, and is excluded from the matched-case intersections (2,180 vs. 2,200 cases).
- **cuTile and TileLang on MI300X.** These do not exist and are shown as N/A in Figures 2 and A1.
- **Reduced B200 TileLang reports.** 13 B200 TileLang NCU reports are reduced (8 with kernel replay) or targeted (5)
  collections. They lack per-opcode or PC-sampling data. Their B200 TMA bytes in Figure A5 (FP16/FP8) are shown as
  "n/c". Six of their A3 cells use static SASS.
- **B200 NCU report coverage.** 9 B200 Triton/cuTile NCU reports exclude auxiliary PyTorch launches (fill/copy) that
  the formal `run()` timing includes. No B200 NCU report records its source commit.
- **Untraced MI300X diagnosis records.** Three MI300X diagnosis records are not fully traced: batch_normalization
  secondary M6; batch_normalization and softmax primary M3; destindex secondary M2.
- **AMD checks not rerun locally.** The raw-dependent AMD QA checks were not rerun locally, because the MI300X raw
  reports are not on this machine. These are `3.report_sha256_recomputed_from_local_reports`,
  `7.derived_metrics_equal_source_kernel_metric_csv` and `prior_analysis_values_reproduced`. Every other AMD check,
  run with `--no-raw`, passed.

## Attribution caveats

- **cuTile versions.** cuTile differs between B200 (1.3.0) and GH200 (1.5.0), and autotune winners may differ between
  devices.
- **TileLang kernel bodies.** On sm_90, TileLang uses fragment-accumulator kernel bodies for 9 operators whose
  Blackwell path uses TMEM. Its GH200 1d_conv kernel body differs from the B200 one.
- **Triton matmul timing.** The Triton matmul-family operators (matmul_fp32_fp16_fp8, matmul_int8, batched_matmul,
  streamk_matmul) cache a transposed B operand outside the timed region.
- **GH200 flash_decode.** cuTile's long-scoreboard stall ratio is *not* higher than Triton's (C 5.5 vs. T 6.2 per
  issue). Only the instruction expansion (3.9×) is attributed.
- **GH200 moe_topk_gating.** The Triton autotune winner runs 64 threads per program, not a single warp.
- **weight_dequant on GH200.** The Triton instruction count is inflated by a different autotune winner (a
  configuration effect, not an architecture effect), which lowers the GH200 ratios to Triton. It is therefore not an
  RQ2 case. Figure A5 keeps it as a marked confounder example (GH200†).
- **MI300X destindex.** The int8 per-lane byte stores are traced from static ISA. weight_dequant and moe_topk_gating
  are controls.

## Figure-specific notes

- **Figure A2.** The selection rule is median within-(device, DSL, dtype) log2-speedup range ≥ 0.5. The variation of
  every candidate is recorded in the manifest.
- **Figures 2 and A1.** The colour scales are clipped (±2 and ±4 in log2). The printed cell values are not clipped.
- **Page geometry.** The figures are designed at 7.0 in (double column) and 3.35 in (single column), with fonts of
  at least 5.5 pt at that size. The ACL template (A4 paper, 2.5 cm margins, 0.6 cm column gap) has a 6.30 in text
  width and a 3.03 in column width. Including a figure at `\textwidth` or `\columnwidth` scales it by about 0.90, so
  the smallest text prints at about 4.95 pt. Re-targeting to 6.30 / 3.03 in only needs `DOUBLE_COL_IN` /
  `SINGLE_COL_IN` in `plot_style.py`, followed by a layout check.

## Not produced

- **RQ4.** Figure 5 and the appendix LLM figure are TODO entries in `plot_manifest.json`. No finalized LLM-generation
  results are part of `artifacts/paper_figures/`. No SOL-efficiency curves, generation costs or human development
  times were fabricated, and no placeholder figure exists.
- **NKI and skill transfer.** NKI/Trainium has no finalized results and is absent from every figure. Skill-transfer
  figures are out of scope.
