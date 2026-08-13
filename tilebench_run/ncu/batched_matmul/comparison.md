# NCU Comparison: batched_matmul

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'BATCH': 32, 'M': 640}` | `{'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUPSIZE': 8, 'num_warps': 4, 'num_stages': 4}` | `{'tile_m': 128, 'tile_n': 128, 'tile_k': 64, 'occupancy': 4, 'group_size': 1}` |
| bf16 | `{'BATCH': 32, 'M': 640}` | `{'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUPSIZE': 1, 'num_warps': 4, 'num_stages': 4}` | `{'tile_m': 128, 'tile_n': 128, 'tile_k': 64, 'occupancy': 4, 'group_size': 1}` |
| fp32 | `{'BATCH': 32, 'M': 640}` | `{'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUPSIZE': 8, 'num_warps': 4, 'num_stages': 3}` | `{'tile_m': 128, 'tile_n': 128, 'tile_k': 64, 'occupancy': 4, 'group_size': 1}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 29.82 us | 43.54 % | 24.05 % | 55.94 % | 36.50 % | 35.15 % | 1.84 Tbyte/s | 128 | 135 register/thread | 0 byte/block | 65.58 Kbyte/block | 3 block / 3 block |
| fp16 | cutile | 30.30 us | 26.62 % | 23.52 % | 33.12 % | 25.55 % | 30.23 % | 1.80 Tbyte/s | 256 | 64 register/thread | 49.43 Kbyte/block | 0 byte/block | 4 block / 4 block |
| bf16 | triton | 29.92 us | 43.76 % | 23.94 % | 55.78 % | 36.07 % | 35.20 % | 1.83 Tbyte/s | 128 | 135 register/thread | 0 byte/block | 65.58 Kbyte/block | 3 block / 3 block |
| bf16 | cutile | 29.79 us | 27.08 % | 24.11 % | 32.91 % | 25.97 % | 30.69 % | 1.85 Tbyte/s | 256 | 64 register/thread | 49.43 Kbyte/block | 0 byte/block | 4 block / 4 block |
| fp32 | triton | 49.50 us | 33.52 % | 33.52 % | 37.64 % | 33.30 % | 37.19 % | 2.57 Tbyte/s | 128 | 135 register/thread | 0 byte/block | 98.35 Kbyte/block | 3 block / 2 block |
| fp32 | cutile | 99.33 us | 51.21 % | 17.46 % | 59.55 % | 16.02 % | 16.89 % | 1.34 Tbyte/s | 256 | 255 register/thread | 180.43 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.02× faster** (29.8 µs vs 30.3 µs).
- **bf16**: cuTile is **1.00× faster** (29.8 µs vs 29.9 µs).
- **fp32**: Triton is **2.01× faster** (49.5 µs vs 99.3 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **bf16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
