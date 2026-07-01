# NCU Comparison: 2d_conv

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3N --launch-count N`, autotune-winner cfg at sweep-max input (`input_size=10240`, kernel 3×3).

**Operator:** single-channel **VALID** 2D correlation (no batch / channels / padding) —
`output[oh,ow] = Σ_{i,j} kernel[i,j] · input[oh+i, ow+j]`. A direct stencil (scalar
weight × shifted input, fp32 accumulate), the 2D analog of `1d_conv` / `3d_conv`.
No im2col, no GEMM, no Tensor Cores.

## Tiling: both backends are 1D-flat (aligned)

Both Triton and cuTile use **1D-flat tiling** (identical to 1d_conv / 3d_conv): each
CTA owns `BLOCK_SIZE`/`TILE` **consecutive flattened output positions**, decodes them
to `(oh, ow)`, and gathers `input[oh+i, ow+j]` with a **1D** index. Valid convolution ⇒
every in-bounds output only touches in-bounds input, so no input mask is needed
(gather padding + masked/OOB-dropped store handle the tail). Same algorithm, same
tiling on both backends — the only remaining difference is DSL codegen.

## Autotune winners (sweep-max, input_size=10240)

| dtype | Triton | cuTile |
|---|---|---|
| fp16 | `{BLOCK_SIZE: 512, num_warps: 4, num_stages: 2}` | `{tile: 1024, occupancy: 8}` |
| fp32 | `{BLOCK_SIZE: 256, num_warps: 4, num_stages: 1}` | `{tile: 1024, occupancy: 4}` |

## Headline (autotune-best @ input_size=10240)

| dtype | Backend | Duration | L1/TEX % | Compute(SM) % | DRAM % | Occupancy % | global-load sectors |
|---|---|---|---|---|---|---|---|
| fp16 | triton | 365 µs | 81.8 | 86.3 | 13.5 | 66.9 | 265 M |
| fp16 | cutile | 376 µs | 40.0 | 79.2 | 13.4 | 46.1 | **90 M** |
| fp32 | triton | 343 µs | 98.3 | 81.7 | 30.5 | 84.4 | 273 M |
| fp32 | cutile | 320 µs | 47.7 | 79.8 | 32.7 | 51.4 | **146 M** |

- **fp16**: ≈ tie — cuTile/Triton = 1.03 (Triton marginally faster).
- **fp32**: cuTile **1.09× faster** (320 vs 343 µs).
- Both compute-bound (SM ≈ 80–86%); neither L1-saturated. cuTile actually issues **fewer**
  global-load sectors (better gather coalescing), Triton has higher occupancy — nets to a tie.

## The 5× catastrophe that was fixed: 2D gather index → 1D gather index

The earlier cuTile version used a **2D block tile** `[BLOCK_R, BLOCK_C]` whose gather index
was a 2D outer product `expand_dims(in_r,1)·s_r + expand_dims(in_c,0)·s_c`. **cuTile 1.3.0's
`ct.gather` does not coalesce a 2D index tile**, so it exploded the L1 traffic:

| cuTile fp16 @ 10240 | OLD (2D-tile gather) | NEW (1D-flat gather) | change |
|---|---|---|---|
| global-load sectors | 495 M | **90 M** | ↓ 5.5× |
| L1/TEX throughput | 98.5 % (saturated) | 40.0 % | no longer L1-bound |
| Compute(SM) | 21.7 % | 79.2 % | now compute-bound |
| Duration | 1710 µs | **376 µs** | ↓ 4.5× |
| fp32 Duration | 1712 µs | **319 µs** | ↓ 5.4× |

Root cause is **cuTile `ct.gather` coalescing depends on the index's dimensionality**:
a 1D index over consecutive positions coalesces; a 2D outer-product index does not.
Triton's `tl.load` coalesced both. Switching cuTile (and then Triton, for a clean
apples-to-apples) to 1D-flat tiling made the gather 1D → coalesced → the gap vanished.

## Cross-operator context (all direct stencils, all now 1D-flat both backends)

| op | gather index | cuTile sectors vs Triton | cuTile L1 | autotune C/T (fp16) |
|---|---|---|---|---|
| 1d_conv | 1D | fewer | ~53% | 1.47 (Triton faster) |
| 3d_conv | 1D | fewer | ~44% | 0.97 (≈ tie) |
| 2d_conv (fixed) | 1D | fewer (90M<265M) | ~40% | 1.03 (≈ tie) |
| 2d_conv (old 2D-tile) | **2D** | **3.2× more (495M)** | **98.5% (bound)** | ~6.4 (5× slower) |

The only case with a 2D gather index (old 2d_conv) was the only case where cuTile was
L1-saturated and multiples slower. With 1D-flat everywhere, all three stencils show
Triton and cuTile within a few percent — a clean DSL-codegen-only comparison.

## Reports

- `triton_fp16.ncu-rep`, `triton_fp32.ncu-rep`
- `cutile_fp16.ncu-rep`, `cutile_fp32.ncu-rep`

## Notes

Both backends: same algorithm (direct stencil), same tiling (1D-flat over flattened
output positions), fp32 accumulation. Open a `.ncu-rep` in `ncu-ui` or
`ncu --import <file> --page details` for per-section detail.
