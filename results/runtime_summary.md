# TileBench Runtime Summary (post-merge wave, exp/timing)

**Branch:** `exp/timing` (= `origin/main` + `fix/2operator_bugs` cherry-pick)
**Hardware:** NVIDIA B200 180GB (dgx003)
**Date:** 2026-05-15
**Method:** mean over all sweep cases per op (case_grid expansion × dtypes; NaN cases excluded)
**Autotune timeout:** 60 minutes per op for `softmax`, `matmul_fp32_fp16_fp8`, `kl_divergence`, `histogramming` (timed out → no autotune data)

## Per-operator mean latency

| Operator | N | Torch (ms) | Triton-def (ms) | cuTile-def (ms) | Triton-tune (ms) | cuTile-tune (ms) | Spd-T def→tune | Spd-C def→tune |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1d_conv | 40 | 2.7898 | 0.6075 | 0.2965 | 0.2945 | 0.2680 | 4.56 → 9.45 | 9.54 → 10.32 |
| 2d_conv | 40 | 0.0547 | 0.7344 | 3.2672 | 0.4628 | 1.1551 | 0.11 → 0.27 | 0.06 → 0.15 |
| 2d_max_pooling | 60 | 0.3892 | 0.0974 | 0.2326 | 0.0887 | 0.2106 | 3.85 → 4.23 | 1.71 → 1.89 |
| 3d_conv | 40 | 1.3733 | 0.0899 | 0.0831 | 0.0875 | 0.0827 | 13.58 → 13.89 | 14.63 → 14.90 |
| argmax | 40 | 0.0359 | 0.0543 | 0.0610 | 0.0199 | 0.0261 | 0.78 → 2.07 | 0.71 → 1.61 |
| batch_normalization | 60 | 0.1960 | 0.0706 | 0.0816 | 0.0582 | 0.0714 | 3.07 → 3.66 | 2.69 → 3.05 |
| batched_matmul | 60 | 0.1200 | 0.0450 | 0.3707 | 0.0212 | 1.9240 | 3.60 → 6.29 | 3.76 → 5.84 |
| bitonic_sort | 40 | 125.5284 | 4.0371 | 3.6743 | 3.5906 | 3.6797 | 33.18 → 37.41 | 36.73 → 36.57 |
| block_sparse_attention | 20 | 16.4389 | 0.0421 | 0.1479 | 0.0421 | 0.1481 | 320.53 → 320.89 | 91.76 → 91.63 |
| cross_entropy | 40 | 0.0118 | 0.0086 | 0.0067 | 0.0044 | 0.0068 | 1.65 → 2.85 | 1.97 → 1.96 |
| dequantize_rowwise | 20 | 0.0477 | 0.0046 | 0.0055 | 0.0049 | 0.0056 | 8.46 → 8.06 | 7.30 → 7.25 |
| destindex | 80 | 0.1058 | 0.1737 | 0.1475 | 0.1478 | 0.1479 | 0.65 → 0.76 | 0.75 → 0.75 |
| dropout | 60 | 0.0612 | 0.0159 | 0.0169 | 0.0160 | 0.0164 | 3.98 → 3.93 | 3.69 → 3.86 |
| flash_attention | 20 | 5.3107 | 14.6163 | 13.6694 | 8.5058 | 6.8264 | 0.34 → 0.58 | 0.37 → 0.71 |
| flash_decode | 20 | 0.0349 | 0.0122 | 0.0439 | 0.0136 | 0.0479 | 3.59 → 3.34 | 1.16 → 1.11 |
| fused_activation | 20 | 0.0531 | 0.0291 | 0.0293 | 0.0292 | 0.0294 | 1.81 → 1.80 | 1.79 → 1.78 |
| gaussian_blur | 40 | 5.6770 | 0.6083 | 1.8579 | 0.6049 | 1.7773 | 9.10 → 9.14 | 3.04 → 3.24 |
| histogramming | 20 | 0.2223 | 0.5314 | 0.5551 | TIMEOUT | TIMEOUT | 0.59 → — | 0.83 → — |
| interleave | 80 | 0.0572 | 0.0165 | 0.0176 | 0.0164 | 0.0175 | 3.67 → 3.76 | 3.52 → 3.53 |
| jacobi_stencil_2d | 60 | 0.5779 | 0.0790 | 0.0665 | 0.0677 | 0.0576 | 6.88 → 8.27 | 8.00 → 9.47 |
| kl_divergence | 20 | 0.0725 | 0.0271 | 0.0261 | TIMEOUT | TIMEOUT | 2.61 → — | 2.67 → — |
| l2_norm | 60 | 0.0753 | 0.0139 | 0.0145 | 0.0128 | 0.0143 | 6.17 → 6.41 | 5.60 → 5.91 |
| layernorm | 60 | 0.0298 | 0.0165 | 0.0225 | 0.0152 | 0.0192 | 1.77 → 1.99 | 1.37 → 1.61 |
| leaky_relu | 60 | 0.0739 | 0.0237 | 0.0242 | 0.0233 | 0.0237 | 2.99 → 3.05 | 2.92 → 3.02 |
| linear_self_attention | 20 | 0.0775 | 1.0034 | 3.8549 | 0.8870 | 3.4412 | 0.27 → 0.46 | 0.08 → 0.14 |
| matmul_fp32_fp16_fp8 | 49 | 2.7687 | 1.5067 | 0.3626 | TIMEOUT | TIMEOUT | 5.07 → — | 11.37 → — |
| matmul_int8 | 20 | 1.8263 | 0.2999 | 0.5165 | 0.0289 | 95.1731 | 6.13 → 7.77 | 3.57 → 0.00 |
| matrix_copy | 80 | 0.0089 | 0.0088 | 0.0089 | 0.0084 | 0.0087 | 1.02 → 1.09 | 1.00 → 1.04 |
| matrix_transpose | 80 | 0.1846 | 0.0369 | 0.0354 | 0.0336 | 0.0346 | 5.57 → 6.05 | 5.47 → 5.69 |
| mean_reduction | 60 | 0.0674 | 0.0146 | 0.0218 | 0.0141 | 0.0152 | 5.44 → 5.63 | 3.65 → 5.11 |
| moe_topk_gating | 60 | 0.0413 | 0.0134 | 0.0151 | 0.0072 | 0.0150 | 3.90 → 6.64 | 3.47 → 3.51 |
| mul2 | 80 | 0.0103 | 0.0102 | 0.0105 | 0.0097 | 0.0103 | 1.01 → 1.10 | 0.98 → 1.01 |
| quantize_global | 20 | 0.0125 | 0.0123 | 0.0124 | 0.0123 | 0.0122 | 1.02 → 1.02 | 1.00 → 1.01 |
| radix_sort | 20 | 0.4484 | 1.1193 | 1.4746 | 1.1062 | 1.4374 | 0.39 → 0.39 | 0.30 → 0.31 |
| relu | 60 | 0.0119 | 0.0113 | 0.0114 | 0.0112 | 0.0112 | 1.06 → 1.07 | 1.04 → 1.06 |
| reverse_array | 80 | 0.0251 | 0.0106 | 0.0108 | 0.0106 | 0.0111 | 2.36 → 2.37 | 2.30 → 2.31 |
| rmsnorm | 60 | 0.1205 | 0.0152 | 0.0164 | 0.0132 | 0.0155 | 8.54 → 9.73 | 7.47 → 8.51 |
| rope | 40 | 0.3529 | 0.0931 | 0.1222 | 0.0573 | 0.1102 | 4.10 → 6.62 | 2.92 → 3.18 |
| sigmoid | 60 | 0.0954 | 0.0246 | 0.0304 | 0.0232 | 0.0303 | 4.32 → 4.73 | 3.18 → 3.26 |
| softmax | 40 | 0.0411 | 0.0182 | 0.0259 | TIMEOUT | TIMEOUT | 2.32 → — | 1.57 → — |
| streamk_matmul | 60 | 2.2414 | 1.6256 | 3.4976 | 2.2207 | 3.4963 | 0.83 → 0.66 | 0.53 → 0.53 |
| swiglu | 60 | 0.0975 | 0.0544 | 0.0702 | 0.0541 | 0.0655 | 1.80 → 1.81 | 1.35 → 1.44 |
| top_k_selection | 20 | 0.0782 | 0.2861 | 0.3975 | 0.2474 | 0.3086 | 0.28 → 0.34 | 0.20 → 0.27 |
| vector_add | 80 | 0.0142 | 0.0140 | 0.0142 | 0.0138 | 0.0142 | 1.02 → 1.03 | 0.99 → 1.00 |
| weight_dequant | 60 | 0.6723 | 0.0368 | 0.1228 | 0.0358 | 0.1166 | 17.75 → 18.09 | 5.69 → 5.72 |

## Legend

- **N** — total sweep cases used to compute mean (case_grid expansion × dtypes; NaN entries excluded)
- **Torch / Triton / cuTile (ms)** — mean per-case kernel latency for the named backend
- **Spd-T / Spd-C** — mean speedup vs Torch (Torch_ms ÷ backend_ms) for Triton / cuTile
- **def → tune** — value moved from default-config mode to autotune mode
- **TIMEOUT** — autotune hit the 60-minute per-op cap; default-mode data still valid

## Notes

- This run uncovered 3 op-level bugs already on `origin/main` (2d_conv missing
  GENERATORS dict entry; linear_self_attention missing module-level
  `_DEFAULT_CONFIG`; bitonic_sort input-generator missing `return`).
  Fixed in PR #87 (`fix/2operator_bugs`) and cherry-picked here so the
  supplemental run could backfill these three ops.
- 45/45 operators now have valid default + autotune data, except the 4 ops
  whose autotune timed out at 60 min (`matmul_fp32_fp16_fp8`, `softmax`,
  `kl_divergence`, `histogramming`) — their default-mode rows are still
  valid; only the autotune column shows TIMEOUT.
- `bitonic_sort`'s Torch reference is the slow PyTorch-op-per-step impl
  (per its docstring, kept as same-algorithm baseline); use `radix_sort`
  for the fast PyTorch sort.
