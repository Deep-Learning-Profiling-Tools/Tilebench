# Algorithm-aware hybrid SOL reference (Figures 2, 3, A1, A2)

The cross-device figures report **proximity to modeled SOL**, `T_SOL / T_k`, instead of speedup over the local PyTorch
baseline. Larger values mean that an implementation reaches a larger fraction of its own device's modeled
limit; 1 is the modeled target. The reference is hybrid (decision H1): published dense compute rates for the direct
GEMM, empirically calibrated rates otherwise; it is neither uniformly empirical nor uniformly hardware-theoretical.
The quantity compares proximity to each accelerator's reference,
not absolute latency, and it is not the "SOL Efficiency" metric of SOL-ExecBench.

Everything here is CPU-only and derived; nothing was recalibrated, rerun or profiled.

## Definition

For operator `o`, DSL `b`, device `d` and input case `c` (dtype and full shape):

```
T_SOL[o,d,c] = max( F / P_peak[mode(o, dtype), d],  Q / BW_peak[d] )      memory-only targets: Q / BW_peak[d]
R[o,b,d,c]   = T_SOL[o,d,c] / T_k[o,b,d,c]                                 never clipped; values above 1 are audited
```

- **F, Q.** The frozen `flops_expr` / `bytes_expr` of `tilebench/benchmarks/operators/<op>/config.yaml` (identical at
  all 315 operator/commit configurations the formal data was measured with), evaluated with the engine's convention:
  `n = params["n"]` or `infer_problem_size`, `dtype_size`, and every numeric case parameter. Two approved overrides of
  `arithmetic_modes.yaml` apply: M3 (causal F of flash_attention) and M4 (compulsory Q of radix_sort), plus the paper
  decision N1: rmsnorm Q = (2n + K)·dtype_size and layernorm Q = (2n + 2K)·dtype_size (compulsory I/O: one read of x and
  of each K-length parameter vector, one write of y; the frozen 3n·dtype_size counted an optional second read).
- **Units.** F in FLOP (OP for INT8 MMA); P in TFLOP/s (TOP/s) times 1e12; BW in GB/s times 1e9; Q in bytes; terms in s
  times 1e3 = ms; `T_k` is the formal autotuned `dsl_ms` of `combined/benchmark_cases_normalized.csv.gz`.
- **Peaks (hybrid policy, decision H1).** `matmul_fp32_fp16_fp8` uses the vendor's published single-GPU dense rate of
  its mode (`sol_modes.DATASHEET_DENSE`: B200 FP16 2250, TF32 1125, FP8 4500; GH200 96 GB FP16 990, TF32 494, FP8 1979;
  MI300X FP16 1307.4, XF32 653.7 TFLOP/s; sources in `sol_modes.DATASHEET_SOURCES`). Every other compute peak and every
  bandwidth come from the PR #323 empirical profiles, read from merge commit
  `72d7cec623e6238643ed5d2099d9aa9889c4ea87` (not merged into this branch). The legacy `peak_performance/<device>.json`
  is never read.

  | device | calibration ID | file sha256 |
  |---|---|---|
  | B200 | `B200-20261006T063415Z-39b55bd3` | `23645505dbb60159c2d7541d09f34366cc5daf9e18feb4242126ac1d39173597` |
  | GH200 | `GH200-20261007T215924Z-0a3f4803-int8layout` | `2bead47d1578ca155cc98b003365961c1638224d9434d731b0d83cc0c0854853` |
  | MI300X | `MI300X-20261009T060617Z-eccfca2c` | `8dfe09c8a1554d1502d9b806563066d52f072cf8278a71ee8d94e3421240edbf` |

- **Aggregation.** GM of case-level R within an operator; GM of operator values within a category; GM of the 45
  operator values (Overall). Cases are never pooled across operators. A cross-device change is
  `log2(R_dev2 / R_dev1)` with both R over the `case_id_v2` valid on both devices.
- **One denominator per task.** `T_SOL` depends on (operator, dtype, case, device) only, so every DSL of a device
  shares it.

## Compute-mode policy (frozen before any SOL ranking was looked at)

The mode is chosen from the operator's frozen canonical algorithm and numerical contract
(`tilebench/llm/v2/contracts/data/<op>/contract.md`: "Algorithm family and structure", "Precision and accumulation").
It is never chosen from the input, accumulator or output dtype alone, from the instructions a compiler generated (the
execution paths of Figure A3 are observations, not inputs), or from benchmark results. A compiler that does not use an
eligible path is measured against that path's peak.

- MMA-eligible matrix algorithms use the matrix peak of their operand format (`fp16_mma`, `bf16_mma`, `fp8_e4m3fn_mma`,
  `int8_mma`; FP32 operands use the TF32 class). FP32 accumulation does not make them vector workloads.
- Non-MMA arithmetic uses the vector peak of the precision the contract requires: `fp16_vector` / `bf16_vector`
  (measured as packed FP16x2 / BF16x2 FMA) when the contract allows input-dtype arithmetic, `fp32_vector` when it
  mandates FP32 arithmetic (for example softmax, normalisation, sigmoid, swiglu, gaussian_blur, int8 dequantization).
- No arithmetic on the data (copies, transposes, indexing, conversion), comparison/selection/sorting work and integer
  arithmetic without a calibrated integer peak use the memory-only target (approved decision M2).
- The decisions agree with the approved per-operator declaration `tilebench/llm/v2/manifests/arithmetic_modes.yaml`
  (revision 2) for all 110 operator/dtype pairs, and the MMA-eligible set equals the operators whose Triton, cuTile and
  TileLang sources all contain a matrix-multiply primitive (`tl.dot`, `ct.mma`, `T.gemm`); the other 35 operators
  contain none in any DSL. `validate_plots.py` (check 24) enforces both.

| class | mode | rows | operator/dtype |
|---|---|---|---|
| MMA-eligible matrix computation | `fp16_mma` | 6 | 1d_conv, 2d_conv, 3d_conv, batched_matmul, matmul_fp32_fp16_fp8, streamk_matmul (fp16) |
| | `bf16_mma` | 2 | batched_matmul, streamk_matmul (bf16) |
| | `tf32_class_mma` | 6 | 1d_conv, 2d_conv, 3d_conv, batched_matmul, matmul_fp32_fp16_fp8, streamk_matmul (fp32) |
| | `fp8_e4m3fn_mma` | 1 | matmul_fp32_fp16_fp8/fp8_e4m3fn (B200, GH200; no MI300X data) |
| | `int8_mma` | 1 | matmul_int8/int8 |
| multistage (MMA + vector/SFU work) | `fp16_mma` | 2 | flash_attention, block_sparse_attention (GEMM FLOPs only; softmax not modeled) |
| | `tf32_class_mma` | 1 | linear_self_attention/fp32 (frozen F at the TF32 rate; per-mode split changes 0 of 60 targets) |
| non-MMA vector / elementwise | `fp16_vector` | 6 | jacobi_stencil_2d, leaky_relu, mul2, rope, vector_add, weight_dequant (fp16) |
| | `bf16_vector` | 5 | jacobi_stencil_2d, leaky_relu, mul2, vector_add, weight_dequant (bf16) |
| | `fp32_vector` | 19 | the same at fp32, rope/fp32; dropout, sigmoid, swiglu (all dtypes); fused_activation, gaussian_blur, dequantize_rowwise/int8 |
| reduction / normalisation (fp32 accumulation) | `fp32_vector` | 21 | batch_normalization, l2_norm, layernorm, mean_reduction, rmsnorm (3 dtypes); cross_entropy, softmax (2); kl_divergence, flash_decode |
| sorting / selection / comparison | `fp32_vector` | 3 | moe_topk_gating (memory-bound in every case, compute/memory <= 0.38) |
| | `memory_only` | 13 | argmax, top_k_selection, bitonic_sort, radix_sort, 2d_max_pooling, relu |
| memory movement / indexing / conversion | `memory_only` | 21 | matrix_copy, matrix_transpose, reverse_array, interleave, destindex (4 dtypes); quantize_global |
| integer arithmetic without MMA | `memory_only` | 3 | mul2/int8, vector_add/int8, histogramming/int32 |

## Decisions taken during this audit (study owner, 2026-10-09)

- **D1, TF32 class on CDNA3: map to `xf32_mma`.** The FP32 GEMM/convolution contracts require 10-bit-mantissa operands
  with FP32 accumulation. Triton 3.6 lowers `input_precision="tf32"` on gfx942 to `v_mfma_f32_*_xf32`; XF32 truncates
  the low 13 mantissa bits and accumulates in FP32. Same precision class as NVIDIA TF32, not bit-identical.
- **D2, MI300X BF16 vector: conditional memory dominance.** MI300X has no calibrated `bf16_vector` (gfx942 has no packed
  BF16 FMA). For the 100 MI300X BF16 cases of leaky_relu, mul2, vector_add, weight_dequant and jacobi_stencil_2d,
  `T_SOL = Q / BW_peak` with status `bf16_vector_peak_unavailable; conditional_memory_dominance`, after verifying for
  every case that its critical throughput `F * BW / Q` is at most 3.99 TFLOP/s (maximum 3.988, jacobi_stencil_2d).
  No BF16 vector peak is assumed and no FP16, FP32 or BF16-MMA rate is substituted. Sensitivity of the RQ1 rows with
  and without these cases: `sol_sensitivity_mi300x_bf16.csv` (MI300X Overall +0.29%, Point-wise +0.34%,
  Stencil/Convolution +1.52%; NVIDIA columns unchanged).
- **D3, memory-only targets: adopt the approved v2 decision M2** for the paper, including int8 mul2/vector_add (no
  device has a calibrated integer vector peak; their compute term would matter only below 3.42 TOP/s). As M2 requires,
  Figure 2 reports the 13 fully memory-only operators beside Overall.

## Coverage

| device | DSL | cases | targets |
|---|---|---|---|
| B200 | Triton, cuTile, TileLang | 2,200 each | 1,460 compute+memory, 740 memory-only |
| GH200 | Triton, cuTile, TileLang | 2,200 each | 1,460 compute+memory, 740 memory-only |
| MI300X | Triton | 2,180 | 1,340 compute+memory, 740 memory-only, 100 conditional memory dominance |

All 45 operators have a target in every column; no operator/dtype lacks a mapping. MI300X has no FP8 E4M3FN rows
(unsupported dtype; its E4M3FNUZ format is a different mode and is never substituted).

## Values above 1

24 of 15,380 case values exceed 1 (no operator or category aggregate does). They are kept, not clipped, and audited in
`sol_above_one_audit.json`:

- **Measurement (swiglu, weight_dequant FP32; 24 cases: swiglu B200 12, GH200 11; weight_dequant B200 Triton 1).**
  Working sets at least 5.3× the L2, at most 2.0% above the calibrated stream-copy bandwidth (max R 1.0198). The copy
  probe is a sustained 1:1 read/write rate, not a bound: swiglu reads two inputs per output, and every case stays below
  the calibrated read-only probe of the same calibration run and below the datasheet HBM bandwidth.

Earlier causes that no longer produce values above 1:
- **rope (61 cases):** remeasured with the input restored before the cache flush (#324, #325, #326).
- **matmul_fp32_fp16_fp8 (127 cases):** scored against the published dense rates (decision H1; max R now 0.84).
- **rmsnorm, layernorm (337 cases):** compulsory-I/O Q (decision N1; max R now 0.92).

## Known limitations of the model

- The empirical peaks are measured sustained rates and the GEMM datasheet peaks are vendor dense specifications;
  neither is a proven bound.
- Simplified operation counts (exponentials, conversions and comparisons are not weighted) only matter for
  compute-bound cases; outside the matrix operators the only one is gaussian_blur.
- The attention operators count only their GEMM FLOPs.
- T_k carries each device's benchmark protocol (warmup, repeats, flush size).

## Files

| file | content |
|---|---|
| `sol_mode_manifest.{json,csv}` | the frozen compute-mode decision per operator and dtype (fields: operator, dtype, canonical_algorithm, arithmetic_precision, mma_eligible, compute_mode, device_specific_peak_key, F/Q expressions, source_file, source_commit, justification, review_status, per-device target status), the decisions and the peak provenance |
| `sol_cases.csv.gz` | one row per device x DSL x valid autotuned case: F, Q, peak, terms, T_SOL, bound, critical throughput, target status, T_k, R |
| `sol_operator_summary.csv` | GM of R per operator, device and DSL, with case counts, bounds and flags |
| `sol_sensitivity_mi300x_bf16.csv` | every RQ1 row with and without the 100 conditional cases |
| `sol_above_one_cases.csv`, `sol_above_one_audit.json` | the cases above 1 and their audited causes |
| `sol_provenance.json` | formula, units, peak hashes, coverage, counts |

Regenerate (CPU only): `python scripts/paper_figures/sol_modes.py && python scripts/paper_figures/sol_data.py`, or the
whole figure build with `build_all_figures.py`.
