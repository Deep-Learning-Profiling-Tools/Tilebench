# Figure captions (drafts)

ACL-style caption drafts for the generated figures. Each caption states what a point, row or cell represents, the
metric, and the main limitation. Numbers in brackets are the values the figure currently shows; they are recomputed
by `validate_plots.py` from `combined/benchmark_cases_normalized.csv.gz`. Abbreviations: T = Triton, C = cuTile,
TL = TileLang, GM = geometric mean.

## Figure 2 (RQ1): `main/fig_rq1_cross_accelerator`

**Figure 2:** Speedup of each tile DSL over the PyTorch baseline on the same accelerator. Each cell is one operator
category (row) for one device/DSL pair (column). For each operator we take the GM of `torch_ms / dsl_ms` over its
valid autotuned input cases. The cell is then the GM over the category's operators; *Overall* covers all 45. Colour
encodes log2 speedup (teal > 1×, red < 1×). cuTile and TileLang do not run on MI300X (hatched N/A). Overall:
B200 T 2.02, C 1.58, TL 1.71; GH200 T 1.85, C 1.53, TL 1.57; MI300X T 1.27. Each device is normalised by its own
PyTorch baseline under its own benchmark protocol, so the figure compares how the DSLs perform relative to vendor
libraries; it does not compare absolute hardware speed. MI300X has 2,180 valid cases instead of 2,200 because FP8
matmul is unsupported there.

## Figure 3 (RQ2): `main/fig_rq2_cross_device_diagnosis`

**Figure 3:** Cross-device diagnosis of six representative input cases. **(A)** Speedup over the local PyTorch
baseline at the exact input case captured by the profiles. The case has the same `case_id_v2` on every device, and
the latency is the formal autotuned latency, not profiler time. There is one row per case and DSL; marker shape
encodes the device and colour encodes the DSL. **(B)** Device-native evidence for the same cases:
- B200 and GH200: Nsight Compute counters and dynamic SASS counts;
- MI300X: static AMDGCN ISA, rocprof-compute metrics and the diagnosis records.

Counters are reported per device and are not comparable across vendors. For GH200 flash_decode only the instruction
expansion is attributed; cuTile's long-scoreboard stall ratio there is not higher than Triton's.

## Figure 4 (RQ3): `main/fig_rq3_within_device_dsl`

**Figure 4:** Comparison of the three NVIDIA tile DSLs on the same device. Each point is one operator, with the
marker giving its category:
- x = cuTile / Triton latency ratio;
- y = TileLang / Triton latency ratio.

Each ratio is a GM over the input cases valid for all three DSLs on that device. Both axes are log2 with shared
limits (1/8×–8×); values below 1 mean faster than Triton. The boxes count the operators for which each DSL has the
lowest GM latency [B200: T 22, TL 18, C 5; GH200: T 26, TL 17, C 2]. They also count how many of those wins are within
5% of the runner-up [B200: 8/8/3; GH200: 14/11/0]; this is not a significance test. B200 and GH200 use different
benchmark protocols, so only within-device ratios are plotted. Labels mark the seven operators farthest from (1×, 1×)
in each panel.

## Appendix Figure A1: `appendix/fig_a1_performance_atlas`

**Figure A1:** Performance atlas of all 45 operators.
- **Left:** the per-operator GM speedup over the local PyTorch baseline for each device/DSL (cell value; colour is
  log2, clipped to 1/16×–16×).
- **Right:** how the relative speedup changes between two devices, shown as 2^Δ with Δ = log2(S_dev2 / S_dev1).
  Both S are computed over the input cases valid on both devices (matched `case_id_v2`). A value above 1 means the
  DSL gains relative to the local PyTorch on the second device.

The right panel is not an absolute hardware speedup.

## Appendix Figure A2: `appendix/fig_a2_shape_dtype`

**Figure A2:** Sensitivity to input shape and data type. Each cell is one input case, coloured by its speedup over the
local PyTorch baseline (log2, clipped at 1/16×–16×). Rows are device/DSL pairs. Columns are the input cases, ordered
by the swept parameter (x label) within groups of a second parameter (group header). For shape sensitivity we show
the operators whose median within-(device, DSL, dtype) log2-speedup range is at least 0.5:
- flash_decode 3.08;
- linear_self_attention 2.63;
- top_k_selection 3.19;
- streamk_matmul 0.73.

For dtype sensitivity we show matmul_fp32_fp16_fp8 with all of its dtypes. vector_add (0.18) is not shown. FP8 E4M3FN
is unsupported on MI300X (N/A row).

## Appendix Figure A3: `appendix/fig_a3_execution_paths`

**Figure A3:** Execution paths observed in the profiles of 15 representative operator/dtype pairs. Cell colour is the
matrix-instruction class: tcgen05, WGMMA, legacy HMMA/IMMA, MFMA, or no matrix instruction. Line 1 of a cell is the
operand load path, taken as the union over the kernels of the profiled run: TMA, cp.async, LDG, TensorDescriptor
lowered to pointer loads, or pointer loads. Line 2 lists staging and flags: SMEM, LDS or TMEM use, register spills,
global or shared atomics at scale, LDSM/STSM, and convert_layout.

Solid cells come from dynamic per-opcode SASS counts (Nsight Compute). Dashed cells come from static SASS or ISA
only, i.e. instructions present in the binary rather than executed counts. This applies to all MI300X cells and to
the six B200 TileLang cells whose reports are reduced collections. If a flag is absent, it was not observed in the available evidence; this does not
prove that the instruction is absent.

## Appendix Figure A4: `appendix/fig_a4_within_device_matrix`

**Figure A4:** Slowdown of each DSL relative to the fastest DSL on the same device, per operator. A cell is the GM
latency of the DSL divided by the lowest GM latency among the three NVIDIA DSLs, over the input cases valid for all
three. The fastest DSL is outlined. Values ≤ 1.05× are drawn neutral as near parity; no significance test is
applied.

## Appendix Figure A5: `appendix/fig_a5_profiling_evidence`

**Figure A5:** Profiling evidence for three mechanisms, at the profiled input case.

**(A) Indexing and reduction overhead.** The dynamic warp-instruction count (`smsp__inst_executed.sum`) of cuTile and
TileLang relative to Triton on the same NVIDIA device. MI300X evidence for these operators is static ISA and appears
as text only. For weight_dequant on GH200 (†), Triton's autotuned configuration differs and inflates its instruction
count; this is a confounder example, not an architecture effect.

**(B) Matrix operand delivery.**
- B200 matmul: TMA load bytes.
- GH200 matmul: dynamic STS instructions per WGMMA, an indicator of operand re-layout through shared memory.
- MI300X: diagnostic latencies (warmup 2 / repeat 10, not the formal protocol) of the TensorDescriptor autotune winner
  against a pointer-load variant. The FP16 variant also changes `num_stages`, so it is not a one-factor experiment.

**(C) Gathered staging and latency hiding.**
- 1d_conv fp16: global-load sectors per request.
- 1d_conv fp16: achieved occupancy (bars) against theoretical occupancy (lines), with issue-active % above the bars.
- MI300X vector_add: diagnostic latencies of load/store cache-modifier variants.

n/c = not collected (reduced TileLang reports). Each numeric axis holds a single vendor's counters.

## Not generated (TODO in `plot_manifest.json`)

- **Figure 5 (RQ4)** and the **appendix LLM figure**: blocked. No finalized LLM-generation results are part of
  `artifacts/paper_figures/`. No SOL-efficiency curves, generation costs or human development times are produced,
  and no placeholder figure is drawn.
- NKI/Trainium results are not finalized and appear in no figure. Skill-transfer figures are out of scope.
