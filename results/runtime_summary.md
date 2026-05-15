# TileBench Runtime Summary

**Branch:** `exp/temp_merge_timing` (main + 40 open PRs merged)
**Hardware:** NVIDIA B200 180GB (dgx003)
**Date:** 2026-05-14
**Method:** mean over all sweep cases per op (case_grid expansion × dtypes)

## Per-operator mean latency

| Operator | N | Torch (ms) | Triton-def (ms) | cuTile-def (ms) | Triton-tune (ms) | cuTile-tune (ms) | Spd-T def→tune | Spd-C def→tune |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1d_conv | 40 | 2.7925 | 0.6087 | 0.2973 | 0.2950 | 0.2687 | 4.55 → 9.44 | 9.51 → 10.28 |
| 2d_conv | 40 | 0.0548 | 0.7342 | 3.2671 | 0.4627 | 1.1554 | 0.11 → 0.27 | 0.06 → 0.15 |
| 2d_max_pooling | 60 | 0.3901 | 0.0977 | 0.2330 | 0.0890 | 0.2110 | 3.85 → 4.22 | 1.71 → 1.89 |
| 3d_conv | 40 | 1.3747 | 0.0902 | 0.0833 | 0.0879 | 0.0830 | 13.56 → 13.85 | 14.61 → 14.84 |
| argmax | 40 | 0.0359 | 0.0543 | 0.0611 | 0.0199 | 0.0260 | 0.78 → 2.07 | 0.71 → 1.62 |
| batch_normalization | 60 | 0.1957 | 0.0707 | 0.0818 | 0.0584 | 0.0716 | 3.06 → 3.65 | 2.68 → 3.04 |
| batched_matmul | 60 | 0.1203 | 0.0451 | 0.3711 | 0.0213 | 2.0035 | 3.60 → 6.29 | 3.77 → 5.92 |
| bitonic_sort | 40 | 125.4965 | 4.0360 | 3.6733 | 3.5897 | 3.6689 | 33.18 → 37.42 | 36.73 → 36.74 |
| block_sparse_attention | 20 | 16.4410 | 0.0422 | 0.1479 | 0.0421 | 0.1481 | 320.49 → 321.07 | 91.77 → 91.66 |
| cross_entropy | 40 | 0.0118 | 0.0086 | 0.0067 | 0.0044 | 0.0068 | 1.65 → 2.87 | 1.97 → 1.97 |
| dequantize_rowwise | 20 | 0.0477 | 0.0046 | 0.0055 | 0.0052 | 0.0056 | 8.48 → 7.83 | 7.31 → 7.29 |
| destindex | 80 | 0.1058 | 0.1738 | 0.1481 | 0.1474 | 0.1479 | 0.65 → 0.76 | 0.75 → 0.75 |
| dropout | 60 | 0.0613 | 0.0159 | 0.0169 | 0.0160 | 0.0164 | 3.98 → 3.93 | 3.70 → 3.85 |
| flash_attention | 20 | 5.3083 | 14.6171 | 13.6691 | 8.4797 | 6.7986 | 0.34 → 0.59 | 0.37 → 0.72 |
| flash_decode | 20 | 0.0349 | 0.0123 | 0.0438 | 0.0136 | 0.0478 | 3.58 → 3.36 | 1.16 → 1.12 |
| fused_activation | 20 | 0.0531 | 0.0291 | 0.0293 | 0.0291 | 0.0294 | 1.81 → 1.81 | 1.79 → 1.79 |
| gaussian_blur | 40 | 5.6772 | 0.6083 | 1.8580 | 0.6049 | 1.7771 | 9.10 → 9.14 | 3.04 → 3.26 |
| histogramming | 20 | 0.2225 | 0.5315 | 0.5552 | TIMEOUT | TIMEOUT | 0.59 → — | 0.83 → — |
| interleave | 80 | 0.0573 | 0.0166 | 0.0176 | 0.0164 | 0.0176 | 3.67 → 3.74 | 3.53 → 3.55 |
| jacobi_stencil_2d | 60 | 0.5785 | 0.0792 | 0.0667 | 0.0679 | 0.0577 | 6.87 → 8.26 | 7.98 → 9.54 |
| kl_divergence | 20 | 0.0726 | 0.0271 | 0.0261 | TIMEOUT | TIMEOUT | 2.61 → — | 2.67 → — |
| l2_norm | 60 | 0.0753 | 0.0139 | 0.0145 | 0.0128 | 0.0146 | 6.17 → 6.44 | 5.60 → 5.95 |
| layernorm | 60 | 0.0299 | 0.0165 | 0.0225 | 0.0153 | 0.0191 | 1.77 → 1.98 | 1.37 → 1.61 |
| leaky_relu | 60 | 0.0740 | 0.0238 | 0.0242 | 0.0234 | 0.0237 | 2.99 → 3.04 | 2.92 → 3.03 |
| linear_self_attention | 15 | 0.0668 | 1.4818 | 1.4655 | 0.3183 | 1.2117 | 0.06 → 0.57 | 0.10 → 0.18 |
| matmul_fp32_fp16_fp8 | 49 | 2.7797 | 1.5056 | 0.3610 | TIMEOUT | TIMEOUT | 5.10 → — | 11.46 → — |
| matmul_int8 | 8 | 0.8141 | 0.1322 | 0.2266 | TIMEOUT | TIMEOUT | 6.21 → — | 3.64 → — |
| matrix_copy | 80 | 0.0090 | 0.0089 | 0.0089 | 0.0084 | 0.0085 | 1.02 → 1.09 | 1.01 → 1.08 |
| matrix_transpose | 32 | 0.0808 | 0.0172 | 0.0166 | 0.0161 | 0.0167 | 5.16 → 5.45 | 5.07 → 5.12 |
| mean_reduction | 60 | 0.0674 | 0.0146 | 0.0218 | 0.0141 | 0.0151 | 5.44 → 5.62 | 3.65 → 5.17 |
| moe_topk_gating | 60 | 0.0415 | 0.0134 | 0.0152 | 0.0072 | 0.0150 | 3.90 → 6.62 | 3.47 → 3.50 |
| mul2 | 80 | 0.0103 | 0.0102 | 0.0105 | 0.0097 | 0.0102 | 1.01 → 1.10 | 0.98 → 1.01 |
| quantize_global | 20 | 0.0125 | 0.0123 | 0.0124 | 0.0123 | 0.0122 | 1.02 → 1.01 | 1.00 → 1.02 |
| radix_sort | 20 | 0.4492 | 1.1237 | 1.4787 | 1.1105 | 1.4414 | 0.39 → 0.39 | 0.30 → 0.31 |
| relu | 60 | 0.0119 | 0.0113 | 0.0114 | 0.0113 | 0.0113 | 1.06 → 1.06 | 1.04 → 1.05 |
| reverse_array | 80 | 0.0251 | 0.0106 | 0.0108 | 0.0105 | 0.0108 | 2.36 → 2.38 | 2.30 → 2.33 |
| rmsnorm | 60 | 0.1205 | 0.0152 | 0.0164 | 0.0132 | 0.0153 | 8.53 → 9.77 | 7.47 → 8.59 |
| rope | 40 | 0.3529 | 0.0930 | 0.1222 | 0.0574 | 0.1103 | 4.10 → 6.59 | 2.92 → 3.17 |
| sigmoid | 60 | 0.0954 | 0.0246 | 0.0304 | 0.0232 | 0.0304 | 4.32 → 4.72 | 3.18 → 3.26 |
| softmax | 40 | 0.0411 | 0.0182 | 0.0259 | TIMEOUT | TIMEOUT | 2.31 → — | 1.57 → — |
| streamk_matmul | 60 | 2.2403 | 1.6266 | 3.4953 | 2.2108 | 3.5225 | 0.83 → 0.66 | 0.53 → 0.52 |
| swiglu | 24 | 0.0440 | 0.0255 | 0.0327 | 0.0254 | 0.0317 | 1.74 → 1.74 | 1.32 → 1.35 |
| top_k_selection | 4 | 0.0387 | 0.1528 | 0.8118 | 0.1343 | 0.8117 | 0.25 → 0.26 | 0.12 → 0.11 |
| vector_add | 80 | 0.0143 | 0.0140 | 0.0142 | 0.0138 | 0.0141 | 1.02 → 1.04 | 0.99 → 1.00 |
| weight_dequant | 60 | 0.6746 | 0.0368 | 0.1231 | 0.0358 | 0.1169 | 17.79 → 18.04 | 5.69 → 5.82 |

## Legend

- **N** — total sweep cases used to compute mean (case_grid expansion × dtypes; NaN entries excluded)
- **Torch / Triton / cuTile (ms)** — mean per-case kernel latency for the named backend
- **Spd-T / Spd-C** — mean speedup vs Torch (Torch_ms ÷ backend_ms) for Triton / cuTile
- **def → tune** — value moved from default-config mode (no autotune) to autotune mode
- **TIMEOUT** — autotune hit the 30-minute per-op cap; default-mode data still valid
  - autotune timed out on: `matmul_fp32_fp16_fp8`, `softmax`, `histogramming`, `kl_divergence`

## Caveats

- `matmul_fp32_fp16_fp8` default reports N=49 instead of 80; some fp16/fp8 cases produced NaN
  (missing kernel paths or numerical overflow) and were excluded from the mean.
- `matmul_int8` autotune produced no valid cuTile timings (NaN); only Triton autotune data is in the autotune JSON.
- `top_k_selection` only swept 4 cases (config has a small case_grid).
- `linear_self_attention` only swept 15 cases.

## Notable findings

**Big cuTile wins** (autotune-mode Spd-C ≥ 5×):

| Op | Spd-C | Spd-T |
|---|---:|---:|
| block_sparse_attention | 91.66 | 321.07 |
| bitonic_sort | 36.74 | 37.42 |
| 3d_conv | 14.84 | 13.85 |
| 1d_conv | 10.28 | 9.44 |
| rmsnorm | 8.59 | 9.77 |
| weight_dequant | 5.82 | 18.04 |
| batched_matmul | 5.92 | 6.29 |
| mean_reduction | 5.17 | 5.62 |
| matrix_transpose | 5.12 | 5.45 |

**Where cuTile underperforms Triton** (Spd-C noticeably below Spd-T):

| Op | Spd-T | Spd-C | Gap |
|---|---:|---:|---:|
| weight_dequant | 18.04 | 5.82 | 3.1× |
| moe_topk_gating | 6.62 | 3.50 | 1.9× |
| batched_matmul | 6.29 | 5.92 (tune fp16+bf16+fp32) | — |
| streamk_matmul | 0.66 | 0.52 | both slower than torch |
| 2d_conv | 0.27 | 0.15 | both slower than torch |
| linear_self_attention | 0.57 | 0.18 | 3.2× |

**Autotune lifted these the most** (def → tune Triton speedup gain):

| Op | Spd-T def | Spd-T tune | gain |
|---|---:|---:|---:|
| linear_self_attention | 0.06 | 0.57 | +9.5× |
| 1d_conv | 4.55 | 9.44 | +2.1× |
| flash_attention | 0.34 | 0.59 | +1.7× |
| moe_topk_gating | 3.90 | 6.62 | +1.7× |
| rope | 4.10 | 6.59 | +1.6× |
| cross_entropy | 1.65 | 2.87 | +1.7× |
| argmax | 0.78 | 2.07 | +2.6× |
| 2d_conv | 0.11 | 0.27 | +2.5× |
