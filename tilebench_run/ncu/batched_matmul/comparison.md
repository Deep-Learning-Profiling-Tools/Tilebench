# NCU Comparison: batched_matmul

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'BATCH': 32, 'M': 640}` | `{'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUPSIZE': 8, 'num_warps': 4, 'num_stages': 4}` | `{'tile_m': 128, 'tile_n': 128, 'tile_k': 64, 'occupancy': 4, 'group_size': 8}` |
| bf16 | `{'BATCH': 32, 'M': 640}` | `{'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUPSIZE': 8, 'num_warps': 4, 'num_stages': 4}` | `{'tile_m': 128, 'tile_n': 128, 'tile_k': 64, 'occupancy': 4, 'group_size': 8}` |
| fp32 | `{'BATCH': 32, 'M': 640}` | `{'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUPSIZE': 8, 'num_warps': 4, 'num_stages': 2}` | `{'tile_m': 128, 'tile_n': 128, 'tile_k': 64, 'occupancy': 4, 'group_size': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 57.70 us | 37.55 % | 12.79 % | 40.69 % | 29.11 % | 40.14 % | 980.62 Gbyte/s | 128 | 51 register/thread | 0 byte/block | 16.40 Kbyte/block | 9 block / 9 block |
| fp16 | cutile | 60.51 us | 29.26 % | 12.23 % | 31.94 % | 26.40 % | 38.83 % | 937.79 Gbyte/s | 256 | 128 register/thread | 98.60 Kbyte/block | 0 byte/block | 2 block / 2 block |
| bf16 | triton | 57.41 us | 37.04 % | 12.83 % | 40.90 % | 29.10 % | 39.61 % | 983.87 Gbyte/s | 128 | 51 register/thread | 0 byte/block | 16.40 Kbyte/block | 9 block / 9 block |
| bf16 | cutile | 60.19 us | 29.06 % | 12.47 % | 32.15 % | 26.57 % | 38.58 % | 956.02 Gbyte/s | 256 | 128 register/thread | 98.60 Kbyte/block | 0 byte/block | 2 block / 2 block |
| fp32 | triton | 293.70 us | 86.54 % | 5.93 % | 91.17 % | 15.00 % | 14.89 % | 455.04 Gbyte/s | 128 | 64 register/thread | 0 byte/block | 32.78 Kbyte/block | 8 block / 6 block |
| fp32 | cutile | 176.29 us | 68.65 % | 10.04 % | 71.61 % | 20.41 % | 22.44 % | 770.11 Gbyte/s | 256 | 128 register/thread | 114.91 Kbyte/block | 0 byte/block | 2 block / 2 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.05× faster** (57.7 µs vs 60.5 µs).
- **bf16**: Triton is **1.05× faster** (57.4 µs vs 60.2 µs).
- **fp32**: cuTile is **1.67× faster** (176.3 µs vs 293.7 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **bf16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — Memory is more heavily utilized than Compute
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
