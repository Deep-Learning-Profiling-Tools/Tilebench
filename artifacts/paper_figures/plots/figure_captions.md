# Figure captions

Publication-ready caption drafts for the generated figures. Every number below is recomputed by
`scripts/paper_figures/validate_plots.py` from the combined data and compared with the figure.

## Figure 2 (RQ1): `main/fig_rq1_cross_accelerator`

**Figure 2:** Speedup of each tile DSL over the PyTorch baseline on the same accelerator.

- **Cells.** Each cell aggregates one workload category (row) for one supported device and DSL (column). We first take
  the geometric mean of `torch_ms / dsl_ms` over each operator's valid autotuned input cases, then the geometric mean
  over the operators in the category, so every operator has equal weight. *Overall* covers all 45 operators. Colour is
  log-scaled and centred at 1×.
- **Overall values.**

  | device | Triton | cuTile | TileLang |
  |---|---|---|---|
  | B200 | 2.02 | 1.58 | 1.71 |
  | GH200 | 1.85 | 1.53 | 1.57 |
  | MI300X | 1.27 | – | – |

- **Comparability.** MI300X has only Triton results. Each device is normalized by its own PyTorch baseline and
  benchmark protocol, so the values describe how each DSL compares with the vendor libraries on that device. They are
  not absolute hardware speedups across GPUs.
- **Coverage.** MI300X has 2,180 valid cases instead of 2,200 because FP8 matmul is not supported there.

## Figure 3 (RQ2): `main/fig_rq2_cross_device_diagnosis`

**Figure 3:** Three mechanisms behind performance changes across devices. Each row uses the single input case captured
by every profile of that operator, with the same `case_id_v2` on all devices.

- **Left panels:** speedup over the local PyTorch baseline from the formal autotuned latency. No profiler duration is
  used.
- **Right panels:** two dynamic Nsight Compute counters for B200 and GH200, summed over the profiled launches.
- **MI300X line:** static ISA or kernel-trace observations. These are not comparable with the NVIDIA counters.

**(A) Matrix operand delivery** (matmul, FP32, M = N = 4096, K = 20480).
- cuTile reaches 1.04× on B200 but 0.61× on GH200.
- On GH200 it executes 21.2 M shared-memory stores, about one per WGMMA instruction. This is consistent with re-laying
  out the B operand in shared memory.
- On B200 its larger tile reads half the TMA bytes of Triton.
- Triton reads a B operand transposed before timing, so part of the difference reflects the benchmark implementation.
- On MI300X, the same TensorDescriptor code is lowered to scalar 32-bit loads.

**(B) Indexing overhead** (destindex, INT8).
- On both NVIDIA devices, cuTile executes about 17× more instructions than Triton and uses 8-bit instead of 128-bit
  stores, while both write the same 2.05 M sectors.
- On MI300X, Triton itself compiles the row copy to per-lane byte stores, and its speedup falls from 4.98× on B200 to
  2.17×. The PyTorch baseline also differs between vendors.

**(C) Memory access and latency hiding** (1d_conv, FP16).
- Load width (16 bit) and DRAM read traffic are the same for both DSLs.
- TileLang touches 7.4× (B200) and 10.4× (GH200) more L1 load sectors than Triton. Its Hopper kernel body differs
  from the Blackwell one.
- The occupancy labels report achieved and theoretical occupancy directly (achieved / theoretical, %). On B200,
  TileLang achieves 6.2% occupancy against a theoretical limit of 18.75%. The other implementations operate close to
  their theoretical limits. The theoretical limit is the maximum resident warp occupancy permitted by the kernel's
  resource and launch configuration, not a predicted value, and the counters do not identify the cause of the
  shortfall.
- Triton's speedup of 1.48× on MI300X comes with a different PyTorch path (MIOpen implicit GEMM plus layout
  transposes). It is not a comparison of the Triton kernels alone.

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

**Figure A1:** Performance of all 45 operators on every supported device and DSL.

- **(A)** Geometric-mean speedup of each operator over the local PyTorch baseline, across its valid autotuned input
  cases. The colour is clipped at 1/16× and 16×; the printed values are not.
- **(B)** Change in relative speedup between two devices, shown as 2^Δ with Δ = log2(S_dev2 / S_dev1). Both S are
  computed over the input cases valid on both devices. A value above 1 means that the DSL gains relative to PyTorch
  on the second device.

Panel B is not an absolute hardware speedup, because it mixes hardware, compiler, library and protocol changes.

## Appendix Figure A2: `appendix/fig_a2_shape_dtype`

**Figure A2:** Sensitivity to input shape and data type.

- **Cells.** Each cell is one input case, coloured by its speedup over the local PyTorch baseline (log scale, clipped
  at 1/16× and 16×). Rows are device and DSL pairs; columns are input cases ordered by the swept parameter, grouped by
  a second parameter where one exists. Panels with more than 12 ungrouped cases label every second case, but every
  case is drawn.
- **Shape sensitivity.** We show the operators whose median within-(device, DSL, dtype) range of log2 speedup is at
  least 0.5: flash_decode (3.08), linear_self_attention (2.63), top_k_selection (3.19) and streamk_matmul (0.73).
- **Dtype sensitivity.** We show matmul with all of its data types. vector_add (0.18) is not shown.
- **Unsupported.** FP8 E4M3FN is not supported on MI300X.

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
