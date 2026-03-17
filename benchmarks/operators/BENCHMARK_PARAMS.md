# Benchmark Parameter Summary

Summary of fixed vs. swept parameters for each operator.

---

## mul2

**Operation:** element-wise `x * 2`

| Parameter | Role | Values |
|-----------|------|--------|
| `n` | swept | 1M, 2M, ..., 20M elements |
| `dtype` | swept | fp16, bf16, fp32, int8 |

No fixed parameters — all inputs are fully described by `n` and `dtype`.

---

## rmsnorm

**Operation:** RMSNorm over the last dimension of a 3D tensor `(batch, M, K)`

| Parameter | Role | Values |
|-----------|------|--------|
| `K` | swept | 512, 1024, ..., 10240 (step 512, 20 points) |
| `dtype` | swept | fp16, bf16, fp32 |
| `batch` | fixed | 1 |
| `M` | fixed | 2048 |

`batch=1, M=2048` mimics a typical LLM prefill batch (2k tokens).
`int8` is excluded because squaring int8 elements overflows and RMSNorm is always applied to floating-point activations in practice.

---

## destindex

**Operation:** scatter copy — `out[dest_loc[i], h, :] = kv[i, h, :]` (MLA KV cache layout)

| Parameter | Role | Values |
|-----------|------|--------|
| `seq_len` | swept | 2048, 4096, ..., 40960 (step 2048, 20 points) |
| `dtype` | swept | fp16, bf16, fp32, int8 |
| `batch_size` | fixed | 1 |
| `kv_nope_head_num` | fixed | 12 |
| `kv_rope_head_num` | fixed | 1 |
| `kv_nope_head_dim` | fixed | 128 |
| `kv_rope_head_dim` | fixed | 64 |

Head configuration matches MLA (Multi-head Latent Attention) typical architecture.
Total tokens per case = `batch_size * seq_len`.

**Note:** cuTile implementation is not available — `ct.store()` does not support
runtime-computed scatter indices. The Triton implementation is fully functional.

---

## relu

**Operation:** element-wise `max(x, 0)`

| Parameter | Role | Values |
|-----------|------|--------|
| `n` | swept | 1M, 2M, ..., 20M elements (step 1M, 20 points) |
| `dtype` | swept | fp16, bf16, fp32, int8 |

No fixed parameters — all inputs are fully described by `n` and `dtype`.

---

## vector_add

**Operation:** element-wise `x + y`

| Parameter | Role | Values |
|-----------|------|--------|
| `n` | swept | 1M, 2M, ..., 20M elements (step 1M, 20 points) |
| `dtype` | swept | fp16, bf16, fp32, int8 |

No fixed parameters — all inputs are fully described by `n` and `dtype`.

---

## dropout

**Operation:** `output = x * x_keep / (1 - p)` (inference-style masked dropout)

| Parameter | Role | Values |
|-----------|------|--------|
| `n` | swept | 1M, 2M, ..., 20M elements (step 1M, 20 points) |
| `dtype` | swept | fp16, bf16, fp32 |
| `p` | fixed | 0.5 |

`int8` is excluded: `x / (1-p)` is undefined for integer types.

---

## swiglu

**Operation:** `output = x * sigmoid(x) * y` (SwiGLU activation, 2D input)

| Parameter | Role | Values |
|-----------|------|--------|
| `N` | swept | 1024, 2048, ..., 8192 (step 1024, 8 points) |
| `dtype` | swept | fp16, bf16, fp32 |
| `M` | fixed | 4096 |

`M=4096` mimics a typical token batch size. `int8` is excluded: `sigmoid(int8)` is undefined.
Total elements per case = `M * N`.

---

## matrix_transpose

**Operation:** `output = x.T` (2D matrix transpose)

| Parameter | Role | Values |
|-----------|------|--------|
| `n` | swept | 1024, 2048, ..., 8192 (step 1024, 8 points) — number of columns |
| `dtype` | swept | fp16, bf16, fp32, int8 |
| `m` | fixed | 4096 — number of rows |

Total elements per case = `m * n`.

---

## divergence_metric

**Operation:** element-wise `(x - y)^2 / (x^2 + eps)` (divergence proxy, output fp32)

| Parameter | Role | Values |
|-----------|------|--------|
| `n` | swept | 1M, 2M, ..., 20M elements (step 1M, 20 points) |
| `dtype` | swept | fp16, bf16, fp32 |
| `eps` | fixed | 1e-6 |

Inputs are read in their native dtype; output is always fp32. `int8` is excluded.

---

## generic_fused_container

**Operation:** `output = relu(x * gate + bias)` (fused pointwise, output fp32)

| Parameter | Role | Values |
|-----------|------|--------|
| `n` | swept | 1M, 2M, ..., 20M elements (step 1M, 20 points) |
| `dtype` | swept | fp16, bf16, fp32 |

Inputs (`x`, `gate`, `bias`) are read in native dtype; output is always fp32. `int8` is excluded.

---

## quantize-global

**Operation:** global cast `fp32 → fp16` (simple dtype conversion / quantization)

| Parameter | Role | Values |
|-----------|------|--------|
| `n` | swept | 1M, 2M, ..., 20M elements (step 1M, 20 points) |
| `dtype` | fixed | fp32 (input); output is always fp16 |

No swept dtype — input is always fp32, output is always fp16.

---

## dequantize-rowwise

**Operation:** global cast `fp16 → fp32` (simple dtype conversion / dequantization)

| Parameter | Role | Values |
|-----------|------|--------|
| `n` | swept | 1M, 2M, ..., 20M elements (step 1M, 20 points) |
| `dtype` | fixed | fp16 (input); output is always fp32 |

No swept dtype — input is always fp16, output is always fp32.

---

## cross_entropy

**Operation:** per-row softmax cross-entropy loss — `loss[i] = -log(softmax(logits[i])[target[i]])`

| Parameter | Role | Values |
|-----------|------|--------|
| `batch_size` | swept | 512, 1024, ..., 10240 (step 512, 20 points) |
| `dtype` | swept | fp16, fp32 |
| `num_classes` | fixed | 512 |

One CTA per row. `BLOCK_CLASSES` is set to `next_power_of_2(num_classes)` at runtime.
Total elements per case = `batch_size * num_classes`.

---

## quantized_gemm

**Operation:** INT8 GEMM with fp32 output — `C = (A_int8 * scale) @ (B_int8 * scale)`

| Parameter | Role | Values |
|-----------|------|--------|
| `k` | swept | 512, 1024, ..., 10240 (step 512, 20 points) — inner dimension |
| `m` | fixed | 2048 — rows of A |
| `n` | fixed | 4096 — columns of B |
| `scale` | fixed | 1.0 |
| `dtype` | fixed | int8 (inputs); output is always fp32 |

Triton autotuner searches over `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, `num_stages` (key: `m, n, k`).
cuTile pads all dimensions to multiples of the tile size; autotune not implemented for cuTile.

---

## layernorm_fwd

**Operation:** LayerNorm over the last dimension — `y = (x - mean) / sqrt(var + eps) * weight + bias`

| Parameter | Role | Values |
|-----------|------|--------|
| `K` | swept | 512, 1024, ..., 10240 (step 512, 20 points) — hidden dimension |
| `dtype` | swept | fp16, bf16, fp32 |
| `batch` | fixed | 1 |
| `M` | fixed | 2048 |

`batch=1, M=2048` mimics a typical LLM prefill batch (2k tokens).
Two-pass algorithm: pass 1 accumulates `sum(x)` and `sum(x²)` to derive mean and variance via `E[x²] − E[x]²`; pass 2 normalizes with weight and bias.
`int8` is excluded: LayerNorm is always applied to floating-point activations in practice.

---

## mean-reduction

**Operation:** Row-wise mean reduction — `out[i] = mean(x[i, :])`

| Parameter | Role | Values |
|-----------|------|--------|
| `N` | swept | 1024, 2048, ..., 20480 (step 1024, 20 points) — reduction dimension |
| `dtype` | swept | fp16, bf16, fp32 |
| `M` | fixed | 2048 — number of rows |

Always benchmarks `dim=1` (row-wise reduction). Output is always float32 (both Triton and cuTile promote input to fp32 before computing; this models the expected use in mixed-precision inference).
Triton: one CTA per row; accumulates in `(BLOCK_M=1, BLOCK_N=1024)` tiles, sums across the tile axis, divides by N.
cuTile: tiled-loop accumulation (same pattern as RMSNorm/LayerNorm); writes one mean value per row to a compact `(M, 1)` output buffer via `ct.sum(..., axis=1) / N`.

---

## argmax

**Operation:** Row-wise argmax over a 2D matrix — `out[i] = argmax(x[i, :])`

| Parameter | Role | Values |
|-----------|------|--------|
| `N` | swept | 1024, 2048, ..., 20480 (step 1024, 20 points) — number of columns |
| `dtype` | swept | fp16, fp32 |
| `M` | fixed | 2048 — number of rows |

Always benchmarks `dim=1` (row-wise reduction), the common case for logit argmax.
Triton: chunked scan (BLOCK_N=128 per iteration) accumulating a running `(best_val, best_idx)` pair per row.
cuTile: single-tile load with `NEG_INF` padding; uses `ct.argmax()` on the padded row; `TILE_SIZE = next_power_of_2(N)`.
Output dtype is `int64` for all backends.
`int8` and `bf16` are excluded: argmax is dtype-agnostic in practice (used on floating-point logits).

---

## l2_norm

**Operation:** L2 normalisation per row — `y = x / ||x||_2`

| Parameter | Role | Values |
|-----------|------|--------|
| `K` | swept | 512, 1024, ..., 10240 (step 512, 20 points) — feature dimension |
| `dtype` | swept | fp16, bf16, fp32 |
| `batch` | fixed | 1 |
| `M` | fixed | 2048 |
| `eps` | fixed | 1e-6 |

`batch=1, M=2048` mimics a typical LLM prefill batch (2k tokens).
Single-pass fused kernel: one CTA per row; `BLOCK_N = next_power_of_2(K)` (capped at 65536/element_size); no weight or bias.
cuTile uses the same tiled two-pass pattern as RMSNorm: pass 1 accumulates `sum(x²)` then `rstd = rsqrt(sum_sq + eps)`; pass 2 applies `y = x * rstd`.
`int8` is excluded: L2 norm is applied to floating-point activations in practice.

---

## conv2d_fwd

**Operation:** 2D convolution forward pass — implicit GEMM (im2col + matmul)

| Parameter | Role | Values |
|-----------|------|--------|
| `H` | swept | 16, 32, ..., 320 (step 16, 20 points) — spatial height = width |
| `dtype` | swept | fp16, fp32 |
| `batch` | fixed | 1 |
| `in_channels` | fixed | 128 |
| `out_channels` | fixed | 128 |
| `kernel_size` | fixed | 3 |
| `stride` | fixed | 1 |
| `padding` | fixed | 1 |
| `groups` | fixed | 1 |

With `padding=1, kernel_size=3, stride=1`, the output spatial size equals the input: `out_H = out_W = H`.
Triton: implicit GEMM kernel — iterates over `(in_channels, kH, kW)` blocks to produce each output tile; autotuner searches `BLOCK_SIZE_BATCH_HEIGHT_WIDTH × BLOCK_SIZE_IN_FEAT × BLOCK_SIZE_OUT_FEAT × num_warps × num_stages`.
cuTile: `torch.nn.functional.unfold` (im2col) followed by cuTile GEMM; groups are processed sequentially.
`int8` is excluded: convolution activations are floating-point in practice.
