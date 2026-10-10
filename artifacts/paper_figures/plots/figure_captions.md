# Figure captions

Publication-ready caption drafts for the generated figures. Every number below is recomputed by
`scripts/paper_figures/validate_plots.py` from the combined data and compared with the figure.

## Figure 2 (RQ1): `main/fig_rq1_cross_accelerator`

**Figure 2:** Proximity to modeled SOL (T_SOL / T_k) of each tile DSL on each accelerator.

- **Metric.** For one operator, input case and device, T_SOL = max(F / P_peak, Q / BW_peak). F and Q are the frozen
  analytical operation count and memory traffic of the operator's algorithm (RMSNorm and LayerNorm use their
  compulsory I/O). P_peak is the peak of the arithmetic mode that the algorithm and its numerical contract require:
  the vendor's published single-GPU dense rate for the direct GEMM (matmul_fp32_fp16_fp8) and the device's empirically
  measured sustained rate (TileArena empirical profiles, PR #323) for every other operator. BW_peak is the measured
  HBM bandwidth. This hybrid SOL reference is neither uniformly empirical nor uniformly hardware-theoretical. T_k is
  the formal autotuned latency. Larger
  values mean the implementation reaches a larger fraction of its own device's modeled limit; 1 is the modeled
  target.
- **Compute mode.** The mode is fixed per operator and data type from the algorithm, not from the input dtype or from
  the instructions a compiler emitted. For example, FP16 `vector_add` uses the FP16 vector peak (measured as packed
  FP16x2 FMA), FP16 `matmul` and `1d_conv` (implicit GEMM) use the FP16 MMA peak, FP32 GEMMs use the TF32-class matrix
  peak (TF32 on NVIDIA, XF32 on MI300X), and FP16 operators whose contract requires FP32 arithmetic use the FP32 vector
  peak. Operators without arithmetic on their data (copies, transposes, indexing), comparison and sorting operators and
  integer operators without a calibrated integer peak use a memory-only target Q / BW_peak.
- **Cells.** We first take the geometric mean of T_SOL / T_k over each operator's valid autotuned input cases, then the
  geometric mean over the operators of a category, so every operator has equal weight. *Overall* covers all 45
  operators; the two rows below it split them into the 13 operators with memory-only targets and the 32 with a compute
  term. Colour is log-scaled from 0.01 to 1.
- **Overall values.**

  | device | Triton | cuTile | TileLang |
  |---|---|---|---|
  | B200 | 0.23 | 0.18 | 0.19 |
  | GH200 | 0.31 | 0.25 | 0.26 |
  | MI300X | 0.18 | – | – |

- **Comparability.** Each column is compared with its own device's hybrid SOL reference. The values compare
  how close each DSL comes to that reference; they do not compare absolute latency or imply identical hardware
  capabilities. MI300X has only Triton results.
- **Coverage.** All 45 operators have a target in every column. MI300X has 2,180 valid cases instead of 2,200 because
  FP8 E4M3FN matmul is not supported there. Its 100 BF16 cases of leaky_relu, mul2, vector_add, weight_dequant and
  jacobi_stencil_2d have no calibrated BF16 vector peak; they use a conditional memory-dominance target, valid because
  their compute term would reach the memory term only below 3.99 TFLOP/s. Without them, MI300X Overall changes by
  +0.29% (largest row change +1.52%, Stencil/Convolution).
- **Values above 1.** No aggregate exceeds 1. 24 of 15,380 case values do (SwiGLU and weight_dequant FP32 at large working sets, at most
  2% above the empirical copy bandwidth); they are kept and audited
  (`sol/sol_above_one_audit.json`).

## Figure 3 (RQ2): `main/fig_rq2_cross_device_diagnosis`

**Figure 3:** Explaining performance differences across accelerators in three cases with distinct mechanisms. Each
row uses the single input case captured by every profile of that operator, with the same `case_id_v2` on all devices.

- **Left panels:** Proximity to modeled SOL (T_SOL / T_k) at that input on every device, from the formal autotuned
  latency and the device's hybrid peaks. The line marks the modeled SOL (1). Targets: (A) the TF32-class MMA term
  (XF32 on MI300X), compute-bound; (B) memory-only; (C) the larger of the FP16 MMA and HBM terms, compute-bound on
  B200 and MI300X and memory-bound on GH200.
- **Right panels** (heading "NVIDIA Profiling (NCU)"): two dynamic Nsight Compute counters for B200 and GH200 only,
  summed over the profiled launches.
- **MI300X line** (labelled ROCm / ISA evidence, ISA evidence, or ISA / rocprof): static ISA or rocprof observations.
  These are not comparable with the NVIDIA counters, so MI300X has no bars.

**(A) Matrix operand delivery** (matmul, FP32, M = N = 4096, K = 20480).
- cuTile reaches 0.67 of the modeled SOL on B200 but 0.54 on GH200; Triton reaches 0.43 and 0.78, and 0.041 on MI300X.
- On GH200 cuTile executes 21.2 M shared-memory stores, about one per WGMMA instruction. This is consistent with
  re-laying out an operand in shared memory.
- On B200 cuTile's larger tile reads half the TMA bytes of Triton.
- Triton reads a B operand transposed before timing, so part of the difference reflects the benchmark implementation.
- On MI300X, the same TensorDescriptor code is lowered to scalar 32-bit loads, with 4.5% MFMA utilization.

**(B) Indexing overhead** (destindex, INT8; memory-only target).
- On both NVIDIA devices, cuTile executes about 17× more instructions than Triton and uses 8-bit instead of 128-bit
  stores, while both write the same 2.05 M sectors. cuTile reaches 0.18 (B200) and 0.29 (GH200), Triton 0.53 and 0.83.
- On MI300X, Triton compiles the row copy to per-lane byte stores (static ISA) and reaches 0.50.

**(C) Memory access and latency hiding** (1d_conv, FP16).
- On both NVIDIA devices, load width (16 bit) and DRAM read traffic are the same for Triton and TileLang.
- TileLang touches 7.4× (B200) and 10.4× (GH200) more L1 load sectors than Triton. Its Hopper kernel body differs
  from the Blackwell one.
- The occupancy chart uses the same encoding as Figure A5: coloured foreground bars show achieved occupancy, grey
  background bars show the theoretical limit, and the labels give achieved / theoretical (%). On B200, TileLang
  achieves 6.2% occupancy against a theoretical limit of 18.75%. The other implementations operate close to their
  theoretical limits. The theoretical limit is the maximum resident warp occupancy permitted by the kernel's resource
  and launch configuration, not a predicted value, and the counters do not identify the cause of the shortfall.
- Triton reaches 0.081 (B200), 0.11 (GH200) and 0.27 (MI300X); TileLang 0.012 and 0.022. On MI300X, 66% of Triton's
  VALU instructions are INT32; no controlled experiment isolates the MI300X difference.

The counters are observations at one input. They are consistent with, but do not prove, the stated mechanisms.

## Figure 4 (RQ3): `main/fig_rq3_within_device_dsl`

**Figure 4:** Comparison of the three NVIDIA DSLs on the same device.

- **Points.** Each point is one operator. x is the cuTile/Triton latency ratio and y is the TileLang/Triton latency
  ratio, each a geometric mean over the input cases that are valid for all three DSLs on that device. Values below 1
  mean faster than Triton. Both panels use the same log-scaled axes.
- **Winner counts.** The panel headers count the operators for which each DSL has the lowest geometric-mean latency:
  B200: Triton 22, TileLang 18, cuTile 5; GH200: Triton 26, TileLang 17, cuTile 2.
  These are numerical winners without uncertainty estimates. The winner is within 5% of the runner-up in
  B200: 8, 8 and 3 and GH200: 14, 11 and 0 of these operators (Triton, TileLang and cuTile).
- **Labels.** Labels mark operators that are at least 2× from Triton on either axis.
- **† Histogramming.** For histogramming, TileLang privatizes the histogram in shared memory, while Triton and cuTile
  update global partial rows atomically. Its position therefore reflects a different algorithm, not better code
  generation for the same algorithm.
- **Protocols.** B200 and GH200 use different benchmark protocols, so only within-device ratios are shown.

## Appendix Figure A1: `appendix/fig_a1_performance_atlas`

**Figure A1:** Proximity to modeled SOL of all 45 operators on every supported device and DSL.

- **(A)** Geometric mean of T_SOL / T_k over each operator's valid autotuned input cases (definition as in Figure 2).
  Colour is log-scaled from 0.001 to 1; amber would mark an operator above the modeled SOL; none is. ‡ marks the MI300X operators whose
  BF16 cases use the conditional memory-dominance target.
- **(B)** Change in proximity between two devices, shown as 2^Δ with Δ = log2(R_dev2 / R_dev1). Both R are computed
  over the input cases valid on both devices. A value above 1 means that the DSL comes closer to its device's modeled
  SOL on the second device.

Panel B is a ratio of normalized SOL proximities, not an absolute accelerator speedup: each R is relative to its own
device's hybrid SOL reference.

## Appendix Figure A2: `appendix/fig_a2_shape_dtype`

**Figure A2:** Sensitivity of the proximity to modeled SOL to input shape and data type.

- **Cells.** Each cell is one input case, coloured by T_SOL / T_k (log scale from 0.001 to 1; amber above 1). Rows are
  device and DSL pairs; columns are input cases ordered by the swept parameter, grouped by a second parameter where one
  exists. Panels with more than 12 ungrouped cases label every second case, but every case is drawn.
- **Panels.** The operators and axes are the same as in the speedup version of this figure: flash_decode,
  linear_self_attention, top_k_selection and streamk_matmul for shape sensitivity, and matmul with all of its data
  types for dtype sensitivity.
- **Missing values.** Grey hatched rows distinguish an unsupported data type (FP8 E4M3FN matmul on MI300X, the only
  case here), an unavailable calibration and a missing measurement; the last two do not occur in these panels.

## Appendix Figure A3: `appendix/fig_a3_execution_paths`

**Figure A3:** Execution paths in the profiles of 15 representative operator and data-type pairs.

- **Cell colour:** the matrix instruction family.
- **Top label:** the operand load path, as the union over the kernels of the profiled run. TMA means TMA operand loads
  (UTMALDG), not TMA stores.
- **Bottom label:** on-chip staging.

**Evidence type.** Solid cells come from dynamic per-opcode SASS counts. Dashed cells come from static SASS or ISA:
instructions present in the binary, not executed counts. This applies to all MI300X cells and to six B200 TileLang
cells whose reports are reduced collections. A missing label means that the path was not observed in the available
evidence; it does not prove that the instruction is absent.

**Secondary flags.** Register spills, global and shared atomics (shared atomics only above 64 per CTA, which excludes
TMEM allocator bookkeeping), LDSM/STSM and layout conversions are listed in `tables/fig_a3_secondary_flags.csv`.

## Appendix Figure A4: `appendix/fig_a4_within_device_matrix`

**Figure A4:** Slowdown of each DSL relative to the fastest DSL on the same device, for all 45 operators.

- **Cells.** A cell is the geometric-mean latency of the DSL divided by the lowest geometric-mean latency among the
  three NVIDIA DSLs, over the input cases valid for all three. The fastest DSL is outlined.
- **Near parity.** Values up to 1.05× are drawn without colour. This is not a significance test. A printed value of
  1.00 without an outline lies within 0.5% of the fastest DSL.

## Appendix Figure A5: `appendix/fig_a5_profiling_evidence`

**Figure A5:** Profiling evidence for the three mechanisms of Figure 3 and for the cases moved out of it. All NVIDIA
values come from one profiled input per operator and data type.

**(A) Executed warp instructions relative to Triton on the same NVIDIA device.** flash_decode runs 16 CTAs on both
devices, and its cuTile kernel executes 2.1× (B200) and 3.9× (GH200) the Triton instructions. For weight_dequant on
GH200 (†), Triton's autotuned configuration differs and inflates its instruction count. This is a confounder example,
not an architecture effect.

**(B) Matrix operand delivery.**
- GH200: dynamic warp-level STS instructions per executed WGMMA instruction, both counted in the same single
  `matmul_kernel` launch. cuTile executes 1.01 (FP32) and 16.05 (FP8, byte-wide STS) per WGMMA but none for FP16.
  Triton executes almost none because it reads a B operand transposed before timing.
- B200: TMA load bytes. n/c marks reduced TileLang reports that did not collect this counter.
- MI300X: diagnostic latencies, not formal measurements (2 warmup and 10 timed runs). The comparison is between the
  TensorDescriptor kernel at its autotuned configuration and a pointer-load variant.
  - FP32: the variant keeps the tile and stage count but also writes C with a masked store.
  - FP16: the variant additionally reduces `num_stages` from 3 to 2 (hatched).
  - Neither comparison isolates the effect of descriptor lowering alone. They show the potential of an alternative
    memory access implementation.

**(C) Memory access and latency hiding.**
- 1d_conv load traffic: global load requests, L1 load sectors and DRAM read bytes, each relative to Triton. All DSLs
  issue 16-bit loads. TileLang touches 7.4× (B200) and 10.4× (GH200) more L1 sectors with the same DRAM traffic, so
  the extra sectors are served by L1 and do not increase DRAM reads.
- 1d_conv occupancy: coloured foreground bars show achieved occupancy, and grey background bars show the theoretical
  occupancy limit permitted by the resource and launch configuration. Labels give achieved over theoretical. On
  B200, TileLang achieves 6.2% against a limit of 18.75%; the other five implementations reach nearly their limits.
  Issue activity, a separate metric, is listed below the axis.
- MI300X vector_add: diagnostic latency of a standalone Triton kernel. With Triton's `.cg` load modifier, gfx942 emits
  `sc0 nt` loads, matching PyTorch under the formal 512 MiB write flush. Without a write flush, the gap from default
  loads shrinks to 1.1×. Store modifiers did not change latency: `.cs` emits `sc0 nt`, `.wt` emits `sc0 sc1`, and `.cg`
  leaves the store unchanged.

No axis mixes vendors, and no axis mixes static with dynamic counts.

## Not generated (TODO entries in `plot_manifest.json`)

- **Figure 5 (RQ4) and the appendix LLM figure.** These are blocked: no finalized LLM generation results are part of
  `artifacts/paper_figures/`. No SOL efficiency curves, generation costs or human development times are produced, and
  no placeholder figure is drawn.
- **NKI/Trainium.** The results are not finalized and appear in no figure. Skill-transfer figures are out of scope.
