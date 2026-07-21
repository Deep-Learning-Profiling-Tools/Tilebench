# TileLang vs Torch Performance Report

This report compares only the TileLang and Torch columns from the no-autotune benchmark run:

`results/runs/20260707_194904_Jul07Run1NoAutotuneNoFullVerification`

The run started on `2026-07-07T23:49:04Z` and finished on `2026-07-08T01:59:16Z` on `dgx003.orc.gmu.edu`. All operator configs in this run have `autotune: false`.

No Triton or cuTile results are used in this report.

## Method

Speedup is computed as:

```text
speedup = torch_ms / tilelang_ms
```

A speedup above `1.0x` means TileLang was faster than Torch. Only rows with numeric Torch and TileLang timings are included in the performance aggregates.

The Torch baseline is the implementation in each operator's `impl_torch.py`. That matters: for example, `bitonic_sort` compares TileLang against a pure PyTorch bitonic-sort baseline, while `radix_sort` and `top_k_selection` compare against optimized Torch primitives.

## Summary

The benchmark requested 45 operators. TileLang produced numeric timings for 41 operators and 2,100 individual cases.

| Metric | Value |
|---|---:|
| Numeric TileLang-vs-Torch cases | 2,100 |
| TileLang faster than Torch | 1,830 cases |
| TileLang slower than Torch | 270 cases |
| TileLang at least 2x faster | 1,238 cases |

At a high level:

- TileLang is strongest when it fuses indexing, type conversion, scaling, reductions, or same-algorithm loop nests into one or a small number of CUDA kernels.
- TileLang is roughly tied with Torch for bandwidth-bound one-pass elementwise kernels such as `mul2`, `vector_add`, `matrix_copy`, and `relu`.
- TileLang loses badly when the Torch baseline maps to a highly optimized library primitive and the TileLang version uses a generic or algorithmically heavier implementation, especially `top_k_selection`, `2d_conv`, `linear_self_attention`, and `radix_sort`.

## Operator Results

| Operator | Cases | Mean speedup | Median | Min | Max | TL faster | TL slower | >=2x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bitonic_sort` | 40 | 35.91x | 35.84x | 31.33x | 40.16x | 40 | 0 | 40 |
| `weight_dequant` | 60 | 18.80x | 19.33x | 7.74x | 30.23x | 60 | 0 | 60 |
| `1d_conv` | 40 | 12.33x | 11.61x | 10.60x | 13.39x | 40 | 0 | 40 |
| `3d_conv` | 40 | 10.12x | 10.73x | 4.09x | 12.41x | 40 | 0 | 40 |
| `rmsnorm` | 60 | 10.02x | 11.39x | 4.73x | 14.79x | 60 | 0 | 60 |
| `jacobi_stencil_2d` | 60 | 9.87x | 9.92x | 2.42x | 17.42x | 60 | 0 | 60 |
| `dequantize_rowwise` | 20 | 8.88x | 8.90x | 5.36x | 14.61x | 20 | 0 | 20 |
| `l2_norm` | 60 | 6.85x | 7.99x | 2.92x | 10.65x | 60 | 0 | 60 |
| `mean_reduction` | 60 | 6.39x | 8.13x | 1.31x | 11.49x | 60 | 0 | 47 |
| `rope` | 40 | 6.35x | 6.23x | 5.94x | 7.72x | 40 | 0 | 40 |
| `gaussian_blur` | 40 | 5.55x | 5.59x | 4.42x | 7.90x | 40 | 0 | 40 |
| `2d_max_pooling` | 60 | 5.51x | 6.38x | 2.15x | 7.42x | 60 | 0 | 60 |
| `moe_topk_gating` | 60 | 4.48x | 3.70x | 2.25x | 12.19x | 60 | 0 | 60 |
| `dropout` | 60 | 4.02x | 4.18x | 2.78x | 5.22x | 60 | 0 | 60 |
| `histogramming` | 20 | 3.90x | 3.82x | 2.87x | 5.06x | 20 | 0 | 20 |
| `matrix_transpose` | 80 | 3.87x | 4.09x | 1.98x | 5.68x | 80 | 0 | 79 |
| `interleave` | 80 | 3.62x | 3.52x | 2.18x | 5.71x | 80 | 0 | 80 |
| `matmul_fp32_fp16_fp8` | 80 | 3.55x | 4.16x | 0.28x | 5.21x | 60 | 20 | 60 |
| `flash_decode` | 20 | 3.36x | 2.45x | 1.43x | 11.67x | 20 | 0 | 12 |
| `sigmoid` | 60 | 3.34x | 4.20x | 0.98x | 4.98x | 58 | 2 | 40 |
| `batch_normalization` | 60 | 3.16x | 3.02x | 1.56x | 10.43x | 60 | 0 | 48 |
| `kl_divergence` | 20 | 2.98x | 2.98x | 1.58x | 4.29x | 20 | 0 | 18 |
| `leaky_relu` | 60 | 2.98x | 3.11x | 2.04x | 3.32x | 60 | 0 | 60 |
| `batched_matmul` | 60 | 2.23x | 1.99x | 1.02x | 5.66x | 60 | 0 | 30 |
| `reverse_array` | 80 | 2.15x | 2.16x | 1.50x | 3.10x | 80 | 0 | 49 |
| `softmax` | 40 | 2.03x | 2.15x | 1.03x | 3.07x | 40 | 0 | 23 |
| `layernorm` | 60 | 1.97x | 1.97x | 1.48x | 2.34x | 60 | 0 | 24 |
| `fused_activation` | 20 | 1.81x | 1.77x | 1.62x | 2.48x | 20 | 0 | 1 |
| `swiglu` | 60 | 1.70x | 1.74x | 1.37x | 1.84x | 60 | 0 | 0 |
| `cross_entropy` | 40 | 1.51x | 1.28x | 0.82x | 4.04x | 31 | 9 | 7 |
| `relu` | 80 | 1.06x | 1.05x | 1.00x | 1.16x | 74 | 6 | 0 |
| `argmax` | 40 | 1.03x | 0.89x | 0.68x | 1.72x | 13 | 27 | 0 |
| `vector_add` | 80 | 1.03x | 1.03x | 1.00x | 1.07x | 77 | 3 | 0 |
| `matrix_copy` | 80 | 1.02x | 1.02x | 0.99x | 1.07x | 69 | 11 | 0 |
| `mul2` | 80 | 1.01x | 1.01x | 0.99x | 1.04x | 65 | 15 | 0 |
| `quantize_global` | 20 | 1.01x | 1.00x | 1.00x | 1.02x | 19 | 1 | 0 |
| `destindex` | 80 | 0.76x | 0.72x | 0.66x | 1.21x | 4 | 76 | 0 |
| `radix_sort` | 20 | 0.19x | 0.20x | 0.14x | 0.21x | 0 | 20 | 0 |
| `linear_self_attention` | 20 | 0.19x | 0.12x | 0.02x | 0.68x | 0 | 20 | 0 |
| `2d_conv` | 40 | 0.15x | 0.13x | 0.07x | 0.38x | 0 | 40 | 0 |
| `top_k_selection` | 20 | 0.07x | 0.07x | 0.04x | 0.11x | 0 | 20 | 0 |

## Dtype Pattern

| Dtype | Cases | Mean speedup | Median | Min | Max | TL faster |
|---|---:|---:|---:|---:|---:|---:|
| `bf16` | 440 | 5.12x | 3.35x | 0.70x | 30.18x | 421 |
| `fp16` | 640 | 5.83x | 3.35x | 0.11x | 36.31x | 568 |
| `fp32` | 780 | 4.04x | 2.13x | 0.02x | 40.16x | 640 |
| `fp8_e4m3fn` | 20 | 5.14x | 5.19x | 4.68x | 5.21x | 20 |
| `fp8_e5m2` | 20 | 5.14x | 5.19x | 4.70x | 5.21x | 20 |
| `int32` | 40 | 2.05x | 2.88x | 0.14x | 5.06x | 20 |
| `int8` | 160 | 2.16x | 1.05x | 0.66x | 5.71x | 141 |

The dtype aggregates mix very different operators, so the operator table is more useful for explaining performance. The fp8 rows are all from `matmul_fp32_fp16_fp8`, where the measured fp8 cases are consistently faster than Torch. The worst fp32 and fp16 rows come from operators where TileLang is algorithmically behind the Torch baseline, not from the dtype alone.

## What The Lowered CUDA Shows

### Bandwidth-bound elementwise kernels stay near Torch

`mul2`, `vector_add`, `matrix_copy`, and `relu` are one-pass global-memory kernels. The lowered CUDA for `mul2` uses one vectorized global load, eight register multiplies, and one vectorized global store:

```cpp
float x_reg[8];
float output_reg[8];
*(ulonglong4*)(x_reg + 0) = tl::load_global_256(...);
#pragma unroll
for (int i = 0; i < 8; ++i) {
  output_reg[i] = (x_reg[i] * 2.0f);
}
tl::store_global_256(...);
```

That is already close to the minimum amount of memory traffic. The result is expected: `mul2` averages `1.01x`, `vector_add` averages `1.03x`, `matrix_copy` averages `1.02x`, and `relu` averages `1.06x`.

The lowered `relu` code is also a simple per-lane branch and store. There is no extra fused work to amortize the memory pass, so it does not create a large gap against Torch.

### `weight_dequant` wins because TileLang fuses scale indexing, conversion, multiply, and store

The Torch reference constructs row and column index tensors, gathers a scale matrix, multiplies, then casts back:

```python
row_idx = torch.arange(M, device=X.device) // TILE_SIZE
col_idx = torch.arange(N, device=X.device) // TILE_SIZE
scale = S[row_idx[:, None], col_idx[None, :]]
result = X.float() * scale.float()
return result.to(X.dtype)
```

The lowered TileLang CUDA instead loads a vector of `X`, loads/broadcasts the scale value, multiplies packed `float2` chunks, and stores the result. This removes intermediate tensor construction and keeps the dequantization path fused. That matches the result: `weight_dequant` averages `18.80x` faster than Torch.

### `bitonic_sort` wins because it compares against the same algorithm in PyTorch

The Torch baseline for `bitonic_sort` is not `torch.sort`; it is a pure PyTorch bitonic implementation with tensor masks, advanced indexing, and host-side loops over bitonic stages.

The lowered TileLang CUDA for a bitonic step is a compact compare-exchange kernel:

```cpp
bool active = (offs < (offs ^ j)) && ((offs ^ j) < padded_n);
if (active) {
  a = work[offs];
  b = work[offs ^ j];
}
bool swap = (((offs & k) == 0) ? (b < a) : (a < b));
if (active) {
  work[offs] = new_a;
  work[offs ^ j] = new_b;
}
```

TileLang avoids the repeated PyTorch tensor expression overhead inside every compare-exchange pass. That explains why `bitonic_sort` is the largest win in the run: mean `35.91x`, max `40.16x`.

### `1d_conv` wins as a direct register-resident convolution

The lowered CUDA for `1d_conv` accumulates several output positions per thread in registers:

```cpp
float acc[8];
for (int k = 0; k < kernel_size; ++k) {
  for (int lane = 0; lane < 8; ++lane) {
    acc[lane] += input[...] * kernel[k];
  }
}
```

For the measured shapes, this direct loop avoids a lot of generic framework overhead and keeps the working set simple. The result is a mean `12.33x` speedup.

### `2d_conv` loses even though it uses tensor-core machinery

The lowered CUDA for `2d_conv` is not a simple scalar implementation. It includes shared memory, async global-to-shared copies, `ldmatrix`, and MMA instructions. The problem is that the generated kernel also carries substantial im2col-style index math and boundary checks, including repeated divisions, modulo operations, and layout transforms.

Torch uses `torch.nn.functional.conv2d` and reaches the cuDNN convolution stack. For these cases, cuDNN beats the generic TileLang explicit-convolution lowering. TileLang averages only `0.15x` of Torch.

### `top_k_selection` loses because TileLang sorts much more than Torch needs

The Torch reference calls:

```python
torch.topk(input.contiguous(), k, largest=True, sorted=True).values
```

The TileLang implementation runs a bitonic-style full sort on a padded array and then returns the first `k` values. The lowered CUDA contains heavy dynamic integer arithmetic around `stride`, `stage`, division, and modulo. This is much more work than a top-k selection primitive needs to do. The result is the worst operator mean in the report: `0.07x`.

### `linear_self_attention` loses because Torch maps the main work to matmul

The Torch implementation computes feature maps and then uses matrix multiplications:

```python
S = phi_k.transpose(0, 1) @ V
Z = phi_k.sum(dim=0)
return (phi_q @ S) / ((phi_q @ Z)[:, None] + eps)
```

The lowered TileLang output kernel loops over the feature dimension inside the kernel, calls scalar `expf`, and accumulates large register arrays such as `numer[128]`. It does not express the main `phi_q @ S` work as a tensor-core GEMM. That explains the `0.19x` mean speedup and the worst single case at `0.024x`.

### `radix_sort` loses against Torch's library sort

The TileLang radix sort path is a multi-kernel radix pipeline with count, prefix-sum, and scatter phases. The lowered scatter phase uses shared-memory scans and guarded global writes. Torch's baseline is `torch.sort(input).values`, which routes to a mature GPU sorting implementation. TileLang averages `0.19x` here.

## TileLang Compile-Lowering Failures

Four requested operators did not produce numeric TileLang timings in this run. They are not counted as speedups or slowdowns above.

| Operator | Cases | TileLang failure |
|---|---:|---|
| `block_sparse_attention` | 20 | Layout infer conflict between `qk` and `p_cast` in a `T.Parallel` loop |
| `flash_attention` | 20 | Layout infer conflict between `scores` and `scores_cast` in a `T.Parallel` loop |
| `matmul_int8` | 20 | Cannot find role for `tirx.Bind` |
| `streamk_matmul` | 60 | `ProducerConsumerWS` failed to replace pipeline loop |

## Best And Worst Individual Cases

Best measured case:

| Operator | Dtype | Shape/config | Torch | TileLang | Speedup |
|---|---|---|---:|---:|---:|
| `bitonic_sort` | `fp32` | `n=500000` | 23.7916 ms | 0.5925 ms | 40.16x |

Worst measured cases:

| Operator | Dtype | Shape/config | Torch | TileLang | Speedup |
|---|---|---|---:|---:|---:|
| `linear_self_attention` | `fp32` | `M=10000, D=256, eps=1e-6` | 0.1308 ms | 5.3626 ms | 0.024x |
| `top_k_selection` | `fp32` | `N=4096, k=16` | 0.0236 ms | 0.5558 ms | 0.043x |
| `2d_conv` | `fp32` | `batch=1, in=128, out=128, kernel=3, H=272` | 0.0778 ms | 1.1237 ms | 0.069x |
| `radix_sort` | `int32` | `n=1000000` | 0.0803 ms | 0.5814 ms | 0.138x |

## Reproducing The Run

From the repository root, the benchmark driver used for this result family is:

```bash
PYTHONPATH=. python scripts/run_bench_all.py --run-name Jul07Run1NoAutotuneNoFullVerification
```

The result artifacts for this report are under:

```text
results/runs/20260707_194904_Jul07Run1NoAutotuneNoFullVerification/
```

The lowered CUDA observations came from TileLang's local cache under:

```text
/home/arustagi/.tilelang/cache/0.1.11-x86_64/kernels/
```

Representative inspected kernels included:

| Operator | Lowered CUDA cache file |
|---|---|
| `mul2` | `028ad227e062b5e149d16869b2a0143dc6fe206693868aec0b90f7825019d9f6/device_kernel.cu` |
| `relu` | `06cad3087f21f092d6035a15e2dc0ed267f513f0ebd8e0d0f86a9a94ba539bb7/device_kernel.cu` |
| `weight_dequant` | `027aa437b6b4b8740be6323903f91596e0ebf9f94b9bf10b836b5f90675c086e/device_kernel.cu` |
| `bitonic_sort` | `1604257d1202304f2df02abd112ab95580ac569c8439f6851c45c26a298dd243/device_kernel.cu` |
| `top_k_selection` | `0a37f78e20bf217af46cd42d8c2109aeabfe1b88e1c1edf957883538357eddce/device_kernel.cu` |
| `1d_conv` | `0d7ea55b08319a8b252194d8608cd982d63313d6c228181a1e457be28018df66/device_kernel.cu` |
| `2d_conv` | `fd4144f63dda08e14e8114b73e4c4728477ec851b0e57c276308b5f89f03ed1a/device_kernel.cu` |

## Caveats

This is a no-autotune comparison. It measures the checked-in/default TileLang configuration against each checked-in Torch reference implementation.

The result is therefore not "TileLang vs the best possible Torch for every mathematical operation." It is "TileLang implementation in this repository vs the Torch baseline in this repository." That distinction is important for operators like `bitonic_sort`, `top_k_selection`, `radix_sort`, and convolutions, where the Torch baseline choice heavily affects the interpretation.
