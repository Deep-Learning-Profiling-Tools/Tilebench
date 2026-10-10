# Limitations of the figures

These limitations apply to the figures in this directory. Device-level entries repeat
`combined/comparison_manifest.json` (`device_limitations`); every figure manifest carries them under
`known_limitations`.

## What the figures can and cannot show

- **Modeled, device-specific target.** Figures 2, 3 (left), A1 and A2 show proximity to modeled SOL (T_SOL / T_k):
  each value is relative to its own device's hybrid SOL reference. They compare how close each DSL comes to
  that reference, not absolute latency or hardware capability (see "Modeled SOL target" below).
- **Cross-device changes.** Figure A1 (B) uses only input cases valid on both devices. A change in proximity still
  combines hardware, compiler and protocol changes and does not isolate any one of them; it is not an accelerator
  speedup.
- **No uncertainty.** Winner counts (Figure 4) and the 5% near-parity band (Figures 4 and A4) have no uncertainty
  analysis; each value is one campaign measurement.
- **Observations, not proofs.** The counters in Figures 3, A3 and A5 describe one profiled input per operator and data
  type. They support the stated mechanisms but do not measure each mechanism's contribution to the formal latency. The
  only controlled experiments are the MI300X diagnostics in A5, and neither of them is a single-factor experiment.

## Modeled SOL target (Figures 2, 3, A1, A2)

Definitions, decisions and tables: `../sol/README.md` and `../sol/sol_mode_manifest.{json,csv}`.

- **Hybrid reference.** matmul_fp32_fp16_fp8 uses the vendor's published single-GPU dense compute rate (decision H1);
  every other P_peak and every BW_peak is a TileArena measured sustained rate (PR #323). Neither kind is a proven
  bound. The empirical matrix peaks are sustained library-GEMM rates (cuBLAS/hipBLASLt or a
  verified Triton probe) at large square shapes, with power-capped clock events in the calibration telemetry.
- **Compute mode from the algorithm.** The mode of each operator and data type is fixed from its frozen contract
  (algorithm and numerical precision), never from the input dtype alone, the compiled ISA or a profiler report
  (Figure A3 shows observed paths and is not used for it). A compiler that does not use an available path is measured
  against that path's peak.
- **TF32 class on MI300X (decision D1).** FP32 GEMM and convolution contracts require 10-bit-mantissa operands with FP32
  accumulation. NVIDIA realises this as TF32, CDNA3 as XF32 (low 13 mantissa bits truncated; Triton maps
  `input_precision="tf32"` to XF32 on gfx942). The two are the same precision class but not bit-identical.
- **Conditional memory dominance (decision D2).** MI300X has no calibrated BF16 vector peak (no packed BF16 FMA on
  gfx942). Its 100 BF16 cases of leaky_relu, mul2, vector_add, weight_dequant and jacobi_stencil_2d use Q / BW_peak,
  which holds for any BF16 vector peak above each case's critical throughput F · BW / Q (at most 3.99 TFLOP/s). No BF16
  peak is assumed or substituted. Without these cases MI300X Overall changes by +0.29%, Point-wise by +0.34% and
  Stencil/Convolution by +1.52% (`../sol/sol_sensitivity_mi300x_bf16.csv`).
- **Memory-only targets (decision M2).** Copies, transposes, indexing, conversion, comparison, sorting and integer
  operators without a calibrated integer peak are scored against Q / BW_peak only. They are not a compute+memory
  roofline, and Figure 2 reports the 13 fully memory-only operators separately beside Overall.
- **Operation counts.** F is the frozen config expression. For non-matrix operators it is mostly a simplified
  per-element count (for example n for vector_add, 6n for layernorm) in which exponentials, conversions and comparisons
  are not weighted, compared with an FMA-based peak that counts two operations per FMA. This only matters for
  compute-bound cases; outside the matrix (MMA) operators the only one is gaussian_blur (FP32 arithmetic, approved
  decision C1). The attention operators count only their GEMM FLOPs; the softmax work has no calibrated mode
  and is not modeled. For linear_self_attention, adding its FP32 vector part at the FP32 vector rate changes no target.
- **Values above 1.** 24 of 15,380 case values exceed 1. They are kept and audited
  (`../sol/sol_above_one_audit.json`): swiglu and weight_dequant FP32 at large working sets are within 2% of the
  calibrated copy rate, which is a sustained 1:1 read/write rate rather than a bound.
- **Protocol effects remain.** T_k still carries each device's benchmark protocol (warmup, repeats, flush size; see
  below), so part of a cross-device change in proximity can come from the protocol.

## Benchmark protocols differ between devices

| device | formal timing | L2/LLC eviction | notes |
|---|---|---|---|
| B200 | warmup 20 / repeat 100 (frozen paper columns) | fixed 64 MB for 41 operators (< 126.5 MB L2) | Triton/cuTile from the paper campaign with cuda-tile 1.3.0 (benchmark source commit not recorded); TileLang measured later (direct runtime, PR #319) in another environment |
| GH200 | warmup 1 / repeat 3 | 120 MiB | CUDA 13.1, cuda-tile 1.5.0 (tileiras 13.4.92), NCU 2025.4.0 |
| MI300X | warmup 20 / repeat 100 | 512 MiB write flush | ROCm eager timing (the HIP-graph request falls back to eager); Triton only |

## Measurement types that stay separate

- **NVIDIA:** Nsight Compute counters and dynamic per-opcode SASS counts (executed instructions).
- **AMD:** static AMDGCN ISA (instructions in the binary), `SQ_INSTS_*` hardware counters, PC samples and ATT hit counts.

Neither figure puts both vendors' counters on one numeric axis.
- In Figure 3, MI300X evidence is a line of text.
- In A5, every axis holds one vendor and one measurement type (`validate_plots.py` checks 13 and 14).
- MI300X diagnostic latencies (2 warmup and 10 timed runs) appear only in A5, on axes titled "(diagnostic)", and never
  in a formal performance figure (check 15).

## Attribution caveats per mechanism

### Matrix operand delivery (Figure 3A, A5 B)

- **Harness asymmetry.** The Triton harness transposes B once and caches the K-major copy outside the timed region
  (`_bt_cache` in `impl_triton.py`). cuTile loads B as [K, N].
- **GH200 cuTile shared-memory stores.** They appear for TF32 (1.01 per WGMMA) and FP8 (16.05 per WGMMA, byte-wide
  STS.U8) but not FP16. This is consistent with re-laying out B in shared memory where WGMMA needs K-major operands. It
  is a hypothesis supported by the counter pattern; no controlled experiment isolates it.
- **Different winners.** The cuTile tiles differ (256×256×64 on B200, 128×128×32 on GH200), and so do the cuda-tile
  versions (1.3.0 vs 1.5.0).
- **MI300X descriptor experiment.** It is not single-factor.
  - FP32: the pointer variant keeps tile and stages but writes C with a masked store.
  - FP16: it also lowers `num_stages` from 3 to 2, because the pointer kernel exceeds LDS at 3 stages.
  - The diagnostic latencies (25.1 → 7.5 ms and 15.2 → 2.3 ms) show the potential of pointer loads, not an isolated
    effect of descriptor lowering.

### Indexing overhead (Figure 3B, A5 A)

- **What is device-independent.** cuTile's 8-bit stores (STG.E.U8) and its 17.3× / 17.8× instruction count appear on
  both NVIDIA devices, so they are not architecture effects.
- **MI300X.** The per-lane byte stores are static ISA. The PyTorch baseline (ATen index_copy) differs by vendor but no
  longer enters Figure 3, whose target is memory-only.
- **weight_dequant on GH200 (†).** A different Triton autotuned configuration inflates its instruction count. It is
  shown as a confounder example.

### Memory access and latency hiding (Figure 3C, A5 C)

- **What sectors per request does and does not mean.** All DSLs issue 16-bit loads (LDG.E.U16, TileLang
  LDG.E.U16.CONSTANT). TileLang's higher sector count per request means that the 32 lanes of a load touch more distinct
  32-byte sectors. Because its L1 hit rate is 92–97% and DRAM read bytes match Triton's (0.67 GB), the extra sectors
  are L1 traffic, not extra DRAM traffic.
- **TileLang kernel bodies.** TileLang uses a different kernel body and configuration on Hopper (block 64×16 vs
  128×64 on B200). Its occupancy reaches the theoretical value on GH200 (18.7%) but not on B200 (6.2% of 18.75%), for
  reasons the counters do not identify.
- **Theoretical occupancy.** This is the maximum resident warp occupancy permitted by the kernel's resource and launch
  configuration (`sm__maximum_warps_per_active_cycle_pct`); it is not a predicted achieved occupancy. A5 draws it as a
  grey background bar behind a narrower achieved bar, and Figure 3C uses the same encoding. The gap between achieved and theoretical occupancy is an observation, not evidence of a specific cause such as
  register pressure, memory stalls or CTA scheduling. Issue activity is a separate metric and is not expressed
  relative to either value.
- **MI300X Triton.** It comes closer to its modeled SOL (0.27) than Triton on B200 (0.081) and GH200 (0.11). No
  controlled experiment explains this; 66% of its VALU instructions are INT32 (index arithmetic). The PyTorch path
  (MIOpen implicit GEMM plus three layout transposes), which explained its speedup in the earlier version, is not part
  of T_SOL / T_k.

### Supporting cases moved out of Figure 3

- **flash_decode (A5 A).** Every device launches 16 CTAs or WGs, so the kernel is latency-bound. cuTile executes 2.1×
  (B200) and 3.9× (GH200) the Triton instructions. On GH200 cuTile's long-scoreboard stall ratio is not higher than
  Triton's, so only the instruction expansion is attributed.
- **block_sparse_attention (A3).** It shows matrix paths and staging only. Its shared-memory footprints stay in
  `combined/figure_evidence.csv` (figure `supporting`).
- **vector_add (A5 C).** On MI300X, loads with the `.cg` modifier emit `sc0 nt` and match PyTorch under the formal
  write flush. The gap from default loads depends on the flush: 1.6× with the 512 MiB write flush, 1.1× with a read
  flush or no flush.
  - The `cache_modifier_ablation_fp32` experiment is documented but not plotted. Its default-store run (131.8 µs) is
    not reproduced by the ISA-identical `.cg`-store run (98.9 µs) or by the launch-configuration sweep (97–105 µs).
  - Store modifiers (`.cs` → `sc0 nt`, `.wt` → `sc0 sc1`, `.cg` → no ISA change) all measured 97.6–98.9 µs.

### Histogramming (Figure 4, A4)

TileLang privatizes the histogram in shared memory, while Triton and cuTile update global partial rows atomically. Its
advantage reflects a different algorithm, not better code generation for the same algorithm.

## Missing or reduced evidence

- **MI300X FP8 matmul.** FP8 E4M3FN matmul is not supported on MI300X (its FP8 format is E4M3FNUZ, a different mode
  that is never substituted). It is shown as N/A (unsupported dtype) in A2 and excluded from the matched intersections
  (2,180 vs. 2,200 cases).
- **cuTile and TileLang on MI300X.** They do not exist and have no column in Figures 2 and A1.
- **Reduced B200 TileLang reports.** 13 B200 TileLang NCU reports are reduced (8, kernel replay) or targeted (5)
  collections without per-opcode or PC-sampling data. FP16/FP8 TMA bytes in A5 are therefore "n/c", and six A3 cells
  use static SASS.
- **B200 report coverage.** Nine B200 Triton/cuTile NCU reports exclude auxiliary PyTorch launches (fill/copy) that
  the formal `run()` timing includes. No B200 NCU report records its source commit.
- **Untraced MI300X diagnosis records.** Three are not fully traced: batch_normalization secondary M6;
  batch_normalization and softmax primary M3; destindex secondary M2.
- **AMD checks not rerun locally.** The raw-dependent AMD QA checks were not rerun locally, because the MI300X raw
  reports are not on this machine:
  - `3.report_sha256_recomputed_from_local_reports`
  - `7.derived_metrics_equal_source_kernel_metric_csv`
  - `prior_analysis_values_reproduced`

## Figure size

All figures are drawn at the ACL text width of 6.30 in. (`acl.sty`: A4 paper, 2.5 cm margins, 0.6 cm column gap;
column width 3.03 in.) At `\textwidth` they print at 1:1, so the font sizes in the manifests are the printed sizes: at
least 7 pt in Figures 2–4 and at least 6 pt in the appendix figures.

## Not produced

- **RQ4.** Figure 5 and the appendix LLM figure are TODO entries in `plot_manifest.json`.
- **NKI.** NKI/Trainium is out of scope.
- **Skill transfer.** Skill-transfer figures are out of scope.
- **No placeholders.** No placeholder data or figure exists.
