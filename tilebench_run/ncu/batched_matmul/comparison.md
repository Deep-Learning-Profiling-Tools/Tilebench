# NCU Comparison: batched_matmul

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'BATCH': 32, 'M': 640}` | `{'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUPSIZE': 8, 'num_warps': 4, 'num_stages': 4}` | `{'tile_m': 128, 'tile_n': 128, 'tile_k': 64, 'occupancy': 4, 'group_size': 1}` |
| bf16 | `{'BATCH': 32, 'M': 640}` | `{'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUPSIZE': 1, 'num_warps': 4, 'num_stages': 4}` | `{'tile_m': 128, 'tile_n': 128, 'tile_k': 64, 'occupancy': 4, 'group_size': 1}` |
| fp32 | `{'BATCH': 32, 'M': 640}` | `{'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUPSIZE': 8, 'num_warps': 4, 'num_stages': 2}` | `{'tile_m': 128, 'tile_n': 128, 'tile_k': 64, 'occupancy': 4, 'group_size': 1}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 29.70 us | 41.28 % | 23.93 % | 51.90 % | 28.29 % | 37.75 % | 1.83 Tbyte/s | 128 | 149 register/thread | 0 byte/block | 65.55 Kbyte/block | 3 block / 3 block |
| fp16 | cutile | 30.05 us | 26.78 % | 23.86 % | 32.62 % | 26.64 % | 30.47 % | 1.83 Tbyte/s | 256 | 64 register/thread | 49.43 Kbyte/block | 0 byte/block | 4 block / 4 block |
| bf16 | triton | 28.93 us | 41.53 % | 24.68 % | 51.86 % | 29.04 % | 38.02 % | 1.89 Tbyte/s | 128 | 149 register/thread | 0 byte/block | 65.55 Kbyte/block | 3 block / 3 block |
| bf16 | cutile | 29.82 us | 26.49 % | 24.07 % | 32.71 % | 25.82 % | 30.07 % | 1.84 Tbyte/s | 256 | 64 register/thread | 49.43 Kbyte/block | 0 byte/block | 4 block / 4 block |
| fp32 | triton | 118.18 us | 68.42 % | 14.66 % | 73.01 % | 40.36 % | 55.18 % | 1.12 Tbyte/s | 128 | 96 register/thread | 0 byte/block | 40.96 Kbyte/block | 5 block / 5 block |
| fp32 | cutile | 18980.00 us | 12.39 % | 8.70 % | 13.32 % | 10.87 % | 5.86 % | 667.38 Gbyte/s | 256 | 255 register/thread | 131.25 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.01× faster** (29.7 µs vs 30.1 µs).
- **bf16**: Triton is **1.03× faster** (28.9 µs vs 29.8 µs).
- **fp32**: Triton is **160.60× faster** (118.2 µs vs 18980.0 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **bf16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
