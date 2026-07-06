# NCU Comparison: streamk_matmul

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'k': 4096, 'm': 8192, 'n': 28672}` | `{'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_warps': 4, 'num_stages': 3}` | `{'tm': 128, 'tn': 128, 'tk': 64, 'group_m': 8, 'occupancy': 16}` |
| bf16 | `{'k': 4096, 'm': 8192, 'n': 28672}` | `{'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 3}` | `{'tm': 128, 'tn': 128, 'tk': 64, 'group_m': 8, 'occupancy': 8}` |
| fp32 | `{'k': 4096, 'm': 8192, 'n': 28672}` | `{'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'num_warps': 4, 'num_stages': 3}` | `{'tm': 128, 'tn': 128, 'tk': 32, 'group_m': 8, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 1962.61 us | 61.89 % | 20.11 % | 62.41 % | 56.90 % | 62.91 % | 1.54 Tbyte/s | 128 | 135 register/thread | 0 byte/block | 98.35 Kbyte/block | 3 block / 2 block |
| fp16 | cutile | 2220.72 us | 72.50 % | 22.14 % | 73.87 % | 55.85 % | 69.00 % | 1.70 Tbyte/s | 256 | 255 register/thread | 229.60 Kbyte/block | 0 byte/block | 1 block / 1 block |
| bf16 | triton | 2073.81 us | 40.44 % | 19.16 % | 41.21 % | 32.91 % | 50.49 % | 1.47 Tbyte/s | 256 | 135 register/thread | 0 byte/block | 147.50 Kbyte/block | 1 block / 1 block |
| bf16 | cutile | 2171.10 us | 71.26 % | 22.86 % | 73.01 % | 57.90 % | 68.07 % | 1.75 Tbyte/s | 256 | 255 register/thread | 229.60 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp32 | triton | 3628.50 us | 61.69 % | 19.15 % | 61.86 % | 60.82 % | 66.36 % | 1.47 Tbyte/s | 128 | 135 register/thread | 0 byte/block | 98.35 Kbyte/block | 3 block / 2 block |
| fp32 | cutile | 3733.42 us | 80.33 % | 21.79 % | 80.81 % | 66.55 % | 80.63 % | 1.67 Tbyte/s | 256 | 255 register/thread | 229.60 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| bf16 | cutile | 1/2 | 571.10 us | `first_wave_Kt1_A2bf16_1v8l0_2t1_3i16_p16_A2bf16_1v` |
| bf16 | cutile | 2/2 | 1600.00 us | `full_tiles_Kt1_A2bf16_1v8l0_2t1_3i16_p16_A2bf16_1v` |
| bf16 | triton | 1/2 | 163.81 us | `first_wave` |
| bf16 | triton | 2/2 | 1910.00 us | `full_tiles` |
| fp16 | cutile | 1/2 | 570.72 us | `first_wave_Kt1_A2f16_1v8l0_2t1_3i16_p16_A2f16_1v8l` |
| fp16 | cutile | 2/2 | 1650.00 us | `full_tiles_Kt1_A2f16_1v8l0_2t1_3i16_p16_A2f16_1v8l` |
| fp16 | triton | 1/2 | 132.61 us | `first_wave` |
| fp16 | triton | 2/2 | 1830.00 us | `full_tiles` |
| fp32 | cutile | 1/2 | 863.42 us | `first_wave_Kt1_A2f32_1v4l0_2t1_3i16_p16_A2f32_1v4l` |
| fp32 | cutile | 2/2 | 2870.00 us | `full_tiles_Kt1_A2f32_1v4l0_2t1_3i16_p16_A2f32_1v4l` |
| fp32 | triton | 1/2 | 218.50 us | `first_wave` |
| fp32 | triton | 2/2 | 3410.00 us | `full_tiles` |

## Key findings (auto-derived)

- **fp16**: Triton is **1.13× faster** (1962.6 µs vs 2220.7 µs).
- **bf16**: Triton is **1.05× faster** (2073.8 µs vs 2171.1 µs).
- **fp32**: Triton is **1.03× faster** (3628.5 µs vs 3733.4 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Compute and Memory are well-balanced
- **bf16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / cutile** — Compute and Memory are well-balanced
- **fp16 / triton** — Compute and Memory are well-balanced
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
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
