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
| fp16 | triton | 83.52 us | 21.65 % | 7.23 % | 23.29 % | 21.64 % | 25.94 % | 554.09 Gbyte/s | 256 | 80 register/thread | 0 byte/block | 36.88 Kbyte/block | 3 block / 3 block |
| fp16 | cutile | 722.21 us | 9.45 % | 1.16 % | 8.82 % | 9.45 % | 12.41 % | 89.17 Gbyte/s | 384 | 168 register/thread | 16.47 Kbyte/block | 0 byte/block | 1 block / 1 block |
| bf16 | triton | 81.06 us | 22.30 % | 7.45 % | 23.25 % | 22.34 % | 25.39 % | 571.34 Gbyte/s | 256 | 80 register/thread | 0 byte/block | 36.88 Kbyte/block | 3 block / 3 block |
| bf16 | cutile | 722.50 us | 9.44 % | 1.16 % | 8.81 % | 9.44 % | 12.38 % | 89.07 Gbyte/s | 384 | 168 register/thread | 16.47 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp32 | triton | 328.64 us | 82.69 % | 4.50 % | 86.16 % | 10.85 % | 11.06 % | 345.11 Gbyte/s | 256 | 114 register/thread | 0 byte/block | 73.74 Kbyte/block | 2 block / 3 block |
| fp32 | cutile | 873.76 us | 8.69 % | 2.73 % | 7.63 % | 8.69 % | 10.21 % | 209.38 Gbyte/s | 384 | 168 register/thread | 32.85 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Key findings (auto-derived)

- **fp16**: Triton is **8.65× faster** (83.5 µs vs 722.2 µs).
- **bf16**: Triton is **8.91× faster** (81.1 µs vs 722.5 µs).
- **fp32**: Triton is **2.66× faster** (328.6 µs vs 873.8 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **bf16 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.33 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.33 full waves across all SMs. Look at Launch Statistics for more details.
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
