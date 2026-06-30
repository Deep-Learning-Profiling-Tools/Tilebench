# NCU Comparison: streamk_matmul

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'k': 4096, 'm': 8192, 'n': 28672}` | `{'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 3}` | `{'tm': 128, 'tn': 128, 'tk': 32, 'group_m': 8, 'occupancy': 16}` |
| bf16 | `{'k': 4096, 'm': 8192, 'n': 28672}` | `{'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 3}` | `{'tm': 128, 'tn': 128, 'tk': 32, 'group_m': 8, 'occupancy': 16}` |
| fp32 | `{'k': 4096, 'm': 8192, 'n': 28672}` | `{'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 3}` | `{'tm': 128, 'tn': 128, 'tk': 32, 'group_m': 8, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 2646.03 us | 68.82 % | 16.61 % | 69.73 % | 49.48 % | 63.02 % | 1.27 Tbyte/s | 256 | 128 register/thread | 0 byte/block | 65.55 Kbyte/block | 2 block / 3 block |
| fp16 | cutile | 14306.14 us | 10.17 % | 2.71 % | 10.20 % | 7.24 % | 12.81 % | 208.10 Gbyte/s | 256 | 255 register/thread | 82.01 Kbyte/block | 0 byte/block | 1 block / 1 block |
| bf16 | triton | 2595.74 us | 67.69 % | 16.97 % | 68.47 % | 50.49 % | 61.75 % | 1.30 Tbyte/s | 256 | 128 register/thread | 0 byte/block | 65.55 Kbyte/block | 2 block / 3 block |
| bf16 | cutile | 14439.38 us | 10.16 % | 2.68 % | 10.21 % | 7.11 % | 12.80 % | 205.27 Gbyte/s | 256 | 255 register/thread | 82.01 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp32 | triton | 17363.72 us | 88.58 % | 3.88 % | 88.89 % | 13.30 % | 12.35 % | 297.47 Gbyte/s | 256 | 125 register/thread | 0 byte/block | 131.09 Kbyte/block | 2 block / 1 block |
| fp32 | cutile | 16225.62 us | 12.80 % | 4.15 % | 11.81 % | 13.02 % | 16.82 % | 318.12 Gbyte/s | 256 | 255 register/thread | 98.40 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| bf16 | cutile | 1/2 | 729.38 us | `first_wave_Kt1_A2bf16_1v8l0_2t1_3i16_p16_A2bf16_1v` |
| bf16 | cutile | 2/2 | 13710.00 us | `full_tiles_Kt1_A2bf16_1v8l0_2t1_3i16_p16_A2bf16_1v` |
| bf16 | triton | 1/4 | 127.10 us | `void at::vectorized_elementwise_kernel<4, at::Fill` |
| bf16 | triton | 2/4 | 114.34 us | `first_wave` |
| bf16 | triton | 3/4 | 2160.00 us | `full_tiles` |
| bf16 | triton | 4/4 | 194.30 us | `void at::vectorized_elementwise_kernel<8, at::bflo` |
| fp16 | cutile | 1/2 | 726.14 us | `first_wave_Kt1_A2f16_1v8l0_2t1_3i16_p16_A2f16_1v8l` |
| fp16 | cutile | 2/2 | 13580.00 us | `full_tiles_Kt1_A2f16_1v8l0_2t1_3i16_p16_A2f16_1v8l` |
| fp16 | triton | 1/4 | 126.66 us | `void at::vectorized_elementwise_kernel<4, at::Fill` |
| fp16 | triton | 2/4 | 114.75 us | `first_wave` |
| fp16 | triton | 3/4 | 2210.00 us | `full_tiles` |
| fp16 | triton | 4/4 | 194.62 us | `void at::vectorized_elementwise_kernel<8, at::floa` |
| fp32 | cutile | 1/2 | 875.62 us | `first_wave_Kt1_A2f32_1v4l0_2t1_3i16_p16_A2f32_1v4l` |
| fp32 | cutile | 2/2 | 15350.00 us | `full_tiles_Kt1_A2f32_1v4l0_2t1_3i16_p16_A2f32_1v4l` |
| fp32 | triton | 1/3 | 126.50 us | `void at::vectorized_elementwise_kernel<4, at::Fill` |
| fp32 | triton | 2/3 | 397.22 us | `first_wave` |
| fp32 | triton | 3/3 | 16840.00 us | `full_tiles` |

## Key findings (auto-derived)

- **fp16**: Triton is **5.41× faster** (2646.0 µs vs 14306.1 µs).
- **bf16**: Triton is **5.56× faster** (2595.7 µs vs 14439.4 µs).
- **fp32**: cuTile is **1.07× faster** (16225.6 µs vs 17363.7 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **bf16 / triton** — Compute and Memory are well-balanced
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — Compute and Memory are well-balanced
- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
