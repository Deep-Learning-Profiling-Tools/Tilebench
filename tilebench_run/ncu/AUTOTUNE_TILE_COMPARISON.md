# Autotune-winner tile/block comparison: Triton vs cuTile

For every (op, dtype) pair that has autotune winners on both backends at the
sweep-max input case, this doc compares the winning **block / tile** dimensions.
This is the same set of configs that the NCU sweep used; the question is
whether Triton and cuTile agree on the optimum tile shape.

## Methodology

Reading `tilebench_run/ncu_catalogue.json`'s
`autotune_winner_per_dtype[<dtype>].{triton,cutile}` cfg dicts. Backend-specific
key names are mapped into shared semantic dims:

| Semantic dim | Triton aliases | cuTile aliases |
|---|---|---|
| `BLOCK_M` | `BLOCK_SIZE_M`, `BLOCK_M` | `tm`, `tile_m`, `TILE_M` |
| `BLOCK_N` | `BLOCK_SIZE_N`, `BLOCK_N` | `tn`, `tile_n`, `TILE_N` |
| `BLOCK_K` | `BLOCK_SIZE_K`, `BLOCK_K` | `tk`, `tile_k`, `TILE_K` |
| `BLOCK_D` | `BLOCK_SIZE_D`, `BLOCK_D` | `td`, `tile_d`, `TILE_D` |
| `BLOCK`   | `BLOCK_SIZE` (1-D ops)    | `tile` |
| `GROUP`   | `GROUP_SIZE_M`, `GROUPSIZE`, `GROUP_SIZE`, `GROUP_M` | `group_size_m`, `group_m`, `group_size` |

Knobs that are intrinsically single-backend (Triton `num_warps`/`num_stages`,
cuTile `occupancy`) are not compared.

## Headline

| Category | # of (op, dtype) pairs | % |
|---|---:|---:|
| ✓ Both backends agree on every shared dim | **22** | 20% |
| ✗ Both expose the same dim but chose different values | **36** | 33% |
| — Not comparable — one side does not autotune a tile dim | **50** | 46% |
| **Total** | **108** | |

## ✓ Full match (22)

Both backends autotune at least one shared tile dim and the winners coincide.
Mostly element-wise / 1-D memory-bound ops where `BLOCK` (the 1-D vector tile)
is the only knob and both autotune sweeps converge on the same value:

| Op / dtype | Shared dim | Winner |
|---|---|---|
| `2d_max_pooling` / fp32 | `BLOCK` | 256 |
| `3d_conv` / fp16 | `BLOCK` | 512 |
| `dropout` / fp16, bf16 | `BLOCK` | 2048 |
| `fused_activation` / fp32 | `BLOCK` | 1024 |
| `gaussian_blur` / fp16, fp32 | `BLOCK` | 512 |
| `matrix_copy` / fp16, bf16, fp32 | `BLOCK` | 1024 |
| `mul2` / fp16, fp32 | `BLOCK` | 1024 |
| `quantize_global` / fp32 | `BLOCK` | 4096 |
| `relu` / fp32 | `BLOCK` | 2048 |
| `reverse_array` / fp16 | `BLOCK` | 2048 |
| `sigmoid` / fp16 | `BLOCK` | 4096 |
| `top_k_selection` / fp32 | `BLOCK` | 512 |
| `vector_add` / bf16, fp32 | `BLOCK` | 1024 |
| `weight_dequant` / fp16, bf16, fp32 | `BLOCK` | 2048 |

## ✗ Differ on at least one shared dim (36)

### Matmul-family ops — most performance-relevant

| Op / dtype | Triton winner | cuTile winner | NCU-measured perf gap (sweep) |
|---|---|---|---|
| **batched_matmul / fp32** | M=**32**, N=128, K=64, GS=**8** | M=**128**, N=128, K=64, GS=**1** | **Triton 160× faster** |
| batched_matmul / fp16 | M=128, N=128, K=**32**, GS=**8** | M=128, N=128, K=**64**, GS=**1** | ~tied |
| batched_matmul / bf16 | M=128, N=128, K=**32**, GS=1 | M=128, N=128, K=**64**, GS=1 | ~tied |
| **matmul_int8 / int8** | M=256, N=64, K=**64**, GS=8 | M=256, N=64, K=**32**, GS=8 | cuTile 1.37× faster |
| **streamk_matmul / fp16** | M=**64**, N=128, K=32, GS=8 | M=**128**, N=128, K=32, GS=8 | Triton 4.21× faster |
| **streamk_matmul / bf16** | M=**64**, N=128, K=32, GS=8 | M=**128**, N=128, K=32, GS=8 | Triton 4.30× faster |
| streamk_matmul / fp32 | M=**64**, N=128, K=32, GS=8 | M=**128**, N=128, K=32, GS=8 | cuTile 2× faster |
| flash_attention / fp16 | M=128, N=**32** | M=128, N=**128** | cuTile 1.28× faster |

The `batched_matmul / fp32` row directly ties into the 160× perf gap NCU
measured: cuTile picks a 128×128 tile (matching its fp16/bf16 winners), but
in fp32 that tile costs 131 KB shmem → only 1 block per SM, killing
occupancy. Triton's autotune correctly picked 32×128 (small M) which keeps
shmem manageable.

### 1-D ops — BLOCK chosen differently

| Op / dtype | Triton BLOCK | cuTile BLOCK |
|---|---:|---:|
| `1d_conv` / fp16, fp32 | 512, 256 | **2048**, **2048** |
| `2d_max_pooling` / fp16, bf16 | 1024 | 256 |
| `3d_conv` / fp32 | 256 | 512 |
| `dropout` / fp32 | 1024 | 2048 |
| `interleave` / fp16, bf16, fp32, int8 | 1024 – 4096 | **8192** (all) |
| `leaky_relu` / fp16, bf16, fp32 | 2048 – 4096 | **8192** (all) |
| `mul2` / bf16, int8 | 2048 | 1024 |
| `relu` / fp16, bf16 | 1024 | 2048 |
| `reverse_array` / bf16, fp32, int8 | 1024 – 8192 | 2048 (all) |
| `sigmoid` / bf16, fp32 | 2048 | 4096 |
| `swiglu` / fp16, bf16, fp32 | 1024 | **4096** (all) |
| `vector_add` / fp16, int8 | 2048 | 1024 |

**Pattern:** cuTile tends to favour **larger** BLOCK (often 4096/8192 — the
top of its search space) on memory-bound ops; Triton converges on smaller
values (1024-2048). Plausible reason: cuTile's fixed-occupancy model
benefits from "fewer, fatter" blocks, while Triton's multi-stage software
pipelining works fine on smaller tiles.

## — Not comparable (50)

For these pairs, at least one backend's autotune search space does **not**
expose any shared tile/block dim — typically only `num_warps`/`num_stages`
(Triton-side) or `occupancy` (cuTile-side) is being tuned, while tile shape
is hard-coded in the impl.

**Neither side autotunes a tile dim** (`occupancy` and/or `num_warps`/`num_stages` only):
- `2d_conv`, `block_sparse_attention`, `cross_entropy`, `destindex` (4 dtypes),
  `flash_decode`, `histogramming`, `jacobi_stencil_2d` (3 dtypes), `layernorm`
  (3 dtypes), `matmul_fp32_fp16_fp8` (4 dtypes), `moe_topk_gating` (3 dtypes),
  `radix_sort`, `rmsnorm` (3 dtypes), `softmax` (2 dtypes)

**Only one side autotunes a tile dim:**
- `argmax` (Triton only — `BLOCK_N`)
- `l2_norm` 3 dtypes (Triton only — `BLOCK_N`)
- `mean_reduction` 3 dtypes (Triton only — `BLOCK_M`, `BLOCK_N`)
- `linear_self_attention` (Triton only — `BLOCK_M`, `BLOCK_D`)
- `batch_normalization` 3 dtypes (cuTile only — `BLOCK`)
- `bitonic_sort` 2 dtypes (cuTile only — `BLOCK`)
- `matrix_transpose` 4 dtypes (cuTile only — `BLOCK`)
- `rope` 2 dtypes (cuTile only — `GROUP`)

## Takeaways

1. **Autotune is necessary, and the winners do diverge.** Naively copying
   a tile shape from one backend's autotune into the other will leave
   performance on the table (or, in matmul-family cases, blow up).
2. **The 160× `batched_matmul / fp32` gap has a tile-shape fingerprint.**
   Triton's autotune correctly responds to dtype (picking smaller M tiles
   for fp32), cuTile's does not — its fp32 winner is the same 128×128 it
   uses for fp16/bf16.
3. **~46% of op-dtype pairs are not block-comparable** because the search
   spaces don't overlap. If we want apples-to-apples tile-shape comparisons
   across the whole benchmark, the autotune search spaces in
   `benchmarks/operators/<op>/impl_{triton,cutile}.py` would need to be
   aligned (add `BLOCK_*` candidates on the side that currently lacks them).
4. **cuTile prefers large 1-D blocks (4-8 KB elements), Triton prefers
   ~2 KB.** Consistent across many element-wise ops; likely reflects the
   different cost models of the two compilers' default codegen.

## Files referenced

- `tilebench_run/ncu_catalogue.json` — autotune winner cfgs
- `tilebench_run/ncu/<op>/comparison.md` — per-op NCU detail
- `tilebench_run/ncu/SUMMARY.md` — global Duration / ratio table
