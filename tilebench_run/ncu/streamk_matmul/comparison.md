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
| fp16 | triton | 3482.72 us | 61.18 % | 19.98 % | 61.50 % | 51.96 % | 75.53 % | 1.53 Tbyte/s | 256 | 64 register/thread | 0 byte/block | 36.88 Kbyte/block | 4 block / 4 block |
| fp16 | cutile | 14664.03 us | 10.13 % | 2.70 % | 10.21 % | 7.29 % | 12.76 % | 207.24 Gbyte/s | 256 | 255 register/thread | 82.01 Kbyte/block | 0 byte/block | 1 block / 1 block |
| bf16 | triton | 3413.13 us | 60.28 % | 20.40 % | 60.56 % | 52.25 % | 74.41 % | 1.57 Tbyte/s | 256 | 64 register/thread | 0 byte/block | 36.88 Kbyte/block | 4 block / 4 block |
| bf16 | cutile | 14662.24 us | 10.15 % | 2.69 % | 10.24 % | 7.08 % | 12.79 % | 206.75 Gbyte/s | 256 | 255 register/thread | 82.01 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp32 | triton | 33424.66 us | 93.70 % | 3.48 % | 93.71 % | 12.31 % | 11.86 % | 266.69 Gbyte/s | 256 | 80 register/thread | 0 byte/block | 73.74 Kbyte/block | 3 block / 3 block |
| fp32 | cutile | 16331.82 us | 12.75 % | 4.13 % | 11.79 % | 12.71 % | 16.84 % | 316.96 Gbyte/s | 256 | 255 register/thread | 98.40 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| bf16 | cutile | 1/4 | 13620.00 us | `full_tiles_kernel_Kt1_A2bf16_1v8l0_2t1_3i16_p16_A2` |
| bf16 | cutile | 2/4 | 195.20 us | `void at::vectorized_elementwise_kernel<8, at::bflo` |
| bf16 | cutile | 3/4 | 126.72 us | `void at::vectorized_elementwise_kernel<4, at::Fill` |
| bf16 | cutile | 4/4 | 720.32 us | `first_wave_kernel_Kt1_A2bf16_1v8l0_2t1_3i16_p16_A2` |
| bf16 | triton | 1/4 | 3010.00 us | `full_tiles` |
| bf16 | triton | 2/4 | 195.10 us | `void at::vectorized_elementwise_kernel<8, at::bflo` |
| bf16 | triton | 3/4 | 127.10 us | `void at::vectorized_elementwise_kernel<4, at::Fill` |
| bf16 | triton | 4/4 | 80.93 us | `first_wave` |
| fp16 | cutile | 1/4 | 13620.00 us | `full_tiles_kernel_Kt1_A2f16_1v8l0_2t1_3i16_p16_A2f` |
| fp16 | cutile | 2/4 | 194.53 us | `void at::vectorized_elementwise_kernel<8, at::floa` |
| fp16 | cutile | 3/4 | 126.62 us | `void at::vectorized_elementwise_kernel<4, at::Fill` |
| fp16 | cutile | 4/4 | 722.88 us | `first_wave_kernel_Kt1_A2f16_1v8l0_2t1_3i16_p16_A2f` |
| fp16 | triton | 1/4 | 3080.00 us | `full_tiles` |
| fp16 | triton | 2/4 | 194.66 us | `void at::vectorized_elementwise_kernel<8, at::floa` |
| fp16 | triton | 3/4 | 126.40 us | `void at::vectorized_elementwise_kernel<4, at::Fill` |
| fp16 | triton | 4/4 | 81.66 us | `first_wave` |
| fp32 | cutile | 1/3 | 875.55 us | `first_wave_kernel_Kt1_A2f32_1v4l0_2t1_3i16_p16_A2f` |
| fp32 | cutile | 2/3 | 15330.00 us | `full_tiles_kernel_Kt1_A2f32_1v4l0_2t1_3i16_p16_A2f` |
| fp32 | cutile | 3/3 | 126.27 us | `void at::vectorized_elementwise_kernel<4, at::Fill` |
| fp32 | triton | 1/3 | 327.97 us | `first_wave` |
| fp32 | triton | 2/3 | 32970.00 us | `full_tiles` |
| fp32 | triton | 3/3 | 126.69 us | `void at::vectorized_elementwise_kernel<4, at::Fill` |

## Key findings (auto-derived)

- **fp16**: Triton is **4.21× faster** (3482.7 µs vs 14664.0 µs).
- **bf16**: Triton is **4.30× faster** (3413.1 µs vs 14662.2 µs).
- **fp32**: cuTile is **2.05× faster** (16331.8 µs vs 33424.7 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **bf16 / triton** — Compute is more heavily utilized than Memory
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — Compute is more heavily utilized than Memory
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
