# NCU Comparison: linear_self_attention

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'eps': 1e-06, 'M': 10000, 'D': 256}` | `{'kv_BLOCK_M': 32, 'kv_BLOCK_N': 32, 'kv_BLOCK_K': 64, 'kv_num_warps': 4, 'kv_num_stages': 3, 'out_BLOCK_M': 64, 'out_BLOCK_N': 128, 'out_BLOCK_K': 32, 'out_num_warps': 8, 'out_num_stages': 3}` | `{'kv_block_m': 128, 'kv_block_n': 32, 'kv_block_k': 64, 'kv_occupancy': 16, 'out_block_m': 64, 'out_block_n': 32, 'out_block_k': 64, 'out_occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 203.30 us | 1.60 % | 0.02 % | 30.87 % | 0.39 % | 0.66 % | 1.36 Gbyte/s | 128 | 32 register/thread | 0 byte/block | 512 byte/block | 16 block / 42 block |
| fp32 | cutile | 498.98 us | 0.83 % | 0.01 % | 21.32 % | 0.17 % | 0.40 % | — | 256 | 255 register/thread | 9.00 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| fp32 | cutile | 1/5 | 8.96 us | `phi_kernel_Kt1_A2f32_1v4l0_2t1_3i16_p16_A2f32_1v4l` |
| fp32 | cutile | 2/5 | 8.13 us | `phi_kernel_Kt1_A2f32_1v4l0_2t1_3i16_p16_A2f32_1v4l` |
| fp32 | cutile | 3/5 | 162.14 us | `kv_gemm_kernel_Kt1_A2f32_1v4l0_2t1_3i16_p16_A2f32_` |
| fp32 | cutile | 4/5 | 274.05 us | `z_kernel_Kt1_A1f32_1i16t1_p16_A2f32_1v4l0_2t1_3i16` |
| fp32 | cutile | 5/5 | 45.70 us | `out_gemm_kernel_Kt1_A2f32_1v4l0_2t1_3i16_p16_A2f32` |
| fp32 | triton | 1/5 | 8.48 us | `phi_kernel` |
| fp32 | triton | 2/5 | 8.48 us | `phi_kernel` |
| fp32 | triton | 3/5 | 56.64 us | `kv_gemm_kernel` |
| fp32 | triton | 4/5 | 115.36 us | `z_kernel` |
| fp32 | triton | 5/5 | 14.34 us | `out_gemm_kernel` |

## Key findings (auto-derived)

- **fp32**: Triton is **2.45× faster** (203.3 µs vs 499.0 µs).

## NCU's own bottleneck verdict

- **fp32 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.05 full waves across all SMs. Look at Launch Statistics for more details.
- **fp32 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.00 full waves across all SMs. Look at Launch Statistics for more details.

## Reports

- `cutile_fp32.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.

## SASS instruction-level findings (manual analysis, updated 2026-08-16)

Both reports above are `--set full` captures at the sweep-max autotune
winners (triton kv 32x32x64 nw4 ns3 / out 64x128x32 nw8 ns3; cuTile
kv 128x32x64 occ16 / out 64x32x64 occ4).

| backend | MMA in SASS | TMA |
|---|---|---|
| torch (cuBLAS TF32) | `UTCHMMA.2CTA` (tcgen05) | UTMALDG/UTMASTG |
| triton | `UTCHMMA` (tcgen05, via host-side TensorDescriptor) | UTMALDG/UTMASTG |
| cuTile | `UTCHMMA` (tcgen05) | UTMALDG/UTMASTG |

The tcgen05 lowering has a **tile_m >= 64 threshold** (tcgen05 MMA
atoms are M=64/128). Controlled probe, same `ct.mma` call, only tile_m
varied:

| tile (MxNxK), tf32 inputs | lowering |
|---|---|
| 16x64x32 | `HMMA.1688.F32.TF32` (legacy mma.sync) |
| 32x64x32 | `HMMA.1688.F32.TF32` (legacy mma.sync) |
| 64x64x32 | `UTCHMMA` (tcgen05) |
| 128x64x32 | `UTCHMMA` (tcgen05) |

The fallback is silent (no diagnostic). An earlier revision of this
section claimed tcgen05-eligible tiles "were in the search space and
lost fairly" — that was incorrect: the #154 GEMM rewrite capped the
cuTile search space at block_m <= 32, so the tuner never saw a
tcgen05-eligible config. With the space widened to mirror Triton's
`_TILE_SHAPES` (block_m/n in {32,64,128}, block_k in {32,64},
occupancy in {4,8,16,32}), the tuner immediately picks block_m=128
for the kv GEMM and block_m=64 for the out GEMM, both lowering to
tcgen05: this rep contains 24 `UTCHMMA` and zero legacy `HMMA`
instructions. Effect: kv_gemm 435.3 -> 162.1 us (2.69x), out_gemm
66.3 -> 45.7 us, end-to-end cuTile 778.6 -> 474.8 us in the CSV
(Triton:cuTile 3.57x -> 2.19x). The earlier "requires split-K"
hypothesis is likewise retired — plain large tiles win outright at
sweep-max (grid drops to 16 CTAs for kv; instruction quality
outweighs CTA parallelism here, the opposite of what that revision
assumed. One search-space entry, 64x32x32 occ8, crashes the tile
compiler (TileCompilerExecutionError rc=5) and is skipped by the
tuner.)

Post-fix residual gap (2.45x NCU / 2.19x CSV): the untuned `z_kernel`
(fixed 32-wide tiles, grid = D/32 = 8 CTAs) now dominates the cuTile
pipeline at 274.1 us (55% of total) vs Triton's 115.4 us — both sides
are severely launch-width-starved on it (8 CTAs on 148 SMs), cuTile
2.4x more so; and kv_gemm remains 2.9x behind Triton's (56.6 us)
despite identical tcgen05 lowering.

Triton's mirror-image precondition, verified on this op: TF32 `tl.dot`
lowers to tcgen05 only on the TMA-descriptor path; the earlier
pointer-based version emitted `HMMA.1688.F32.TF32` and no TMA
(2.1x slower end-to-end at sweep-max).
