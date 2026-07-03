# NCU Comparison: streamk_matmul

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'k': 4096, 'm': 8192, 'n': 28672}` | `{'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 3}` | `{'tm': 128, 'tn': 128, 'tk': 64, 'group_m': 8, 'occupancy': 16}` |
| bf16 | `{'k': 4096, 'm': 8192, 'n': 28672}` | `{'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 3}` | `{'tm': 128, 'tn': 128, 'tk': 64, 'group_m': 8, 'occupancy': 16}` |
| fp32 | `{'k': 4096, 'm': 8192, 'n': 28672}` | `{'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 3}` | `{'tm': 128, 'tn': 128, 'tk': 64, 'group_m': 8, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 1977.57 us | 61.80 % | 19.59 % | 62.16 % | 54.60 % | 63.32 % | 1.50 Tbyte/s | 256 | 71 register/thread | 0 byte/block | 98.35 Kbyte/block | 3 block / 2 block |
| fp16 | cutile | 2269.41 us | 72.75 % | 21.59 % | 74.50 % | 56.00 % | 69.04 % | 1.66 Tbyte/s | 256 | 255 register/thread | 229.60 Kbyte/block | 0 byte/block | 1 block / 1 block |
| bf16 | triton | 1938.18 us | 59.73 % | 20.06 % | 59.89 % | 55.65 % | 61.32 % | 1.54 Tbyte/s | 256 | 71 register/thread | 0 byte/block | 98.35 Kbyte/block | 3 block / 2 block |
| bf16 | cutile | 2209.25 us | 72.00 % | 22.36 % | 73.56 % | 58.30 % | 68.60 % | 1.72 Tbyte/s | 256 | 255 register/thread | 229.60 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp32 | triton | 3681.12 us | 60.57 % | 18.45 % | 61.98 % | 58.60 % | 65.77 % | 1.42 Tbyte/s | 256 | 71 register/thread | 0 byte/block | 98.35 Kbyte/block | 3 block / 2 block |
| fp32 | cutile | 3895.42 us | 67.16 % | 19.67 % | 68.32 % | 60.36 % | 68.17 % | 1.51 Tbyte/s | 256 | 255 register/thread | 213.19 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| bf16 | cutile | 1/2 | 569.25 us | `first_wave_Kt1_A2bf16_1v8l0_2t1_3i16_p16_A2bf16_1v` |
| bf16 | cutile | 2/2 | 1640.00 us | `full_tiles_Kt1_A2bf16_1v8l0_2t1_3i16_p16_A2bf16_1v` |
| bf16 | triton | 1/2 | 98.18 us | `first_wave` |
| bf16 | triton | 2/2 | 1840.00 us | `full_tiles` |
| fp16 | cutile | 1/2 | 569.41 us | `first_wave_Kt1_A2f16_1v8l0_2t1_3i16_p16_A2f16_1v8l` |
| fp16 | cutile | 2/2 | 1700.00 us | `full_tiles_Kt1_A2f16_1v8l0_2t1_3i16_p16_A2f16_1v8l` |
| fp16 | triton | 1/2 | 97.57 us | `first_wave` |
| fp16 | triton | 2/2 | 1880.00 us | `full_tiles` |
| fp32 | cutile | 1/2 | 675.42 us | `first_wave_Kt1_A2f32_1v4l0_2t1_3i16_p16_A2f32_1v4l` |
| fp32 | cutile | 2/2 | 3220.00 us | `full_tiles_Kt1_A2f32_1v4l0_2t1_3i16_p16_A2f32_1v4l` |
| fp32 | triton | 1/2 | 181.12 us | `first_wave` |
| fp32 | triton | 2/2 | 3500.00 us | `full_tiles` |

## Key findings (auto-derived)

- **fp16**: Triton is **1.15× faster** (1977.6 µs vs 2269.4 µs).
- **bf16**: Triton is **1.14× faster** (1938.2 µs vs 2209.2 µs).
- **fp32**: Triton is **1.06× faster** (3681.1 µs vs 3895.4 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Compute and Memory are well-balanced
- **bf16 / triton** — Compute and Memory are well-balanced
- **fp16 / cutile** — Compute and Memory are well-balanced
- **fp16 / triton** — Compute and Memory are well-balanced
- **fp32 / cutile** — Compute and Memory are well-balanced
- **fp32 / triton** — Compute and Memory are well-balanced

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
