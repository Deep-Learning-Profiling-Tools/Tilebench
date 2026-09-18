# NCU Comparison: batched_matmul

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'BATCH': 32, 'M': 640}` | `{'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUPSIZE': 1, 'num_warps': 4, 'num_stages': 4}` | `{'tile_m': 128, 'tile_n': 128, 'tile_k': 64, 'occupancy': 4, 'group_size': 1}` |
| bf16 | `{'BATCH': 32, 'M': 640}` | `{'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUPSIZE': 1, 'num_warps': 4, 'num_stages': 4}` | `{'tile_m': 128, 'tile_n': 128, 'tile_k': 64, 'occupancy': 4, 'group_size': 1}` |
| fp32 | `{'BATCH': 32, 'M': 640}` | `{'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUPSIZE': 8, 'num_warps': 4, 'num_stages': 3}` | `{'tile_m': 128, 'tile_n': 128, 'tile_k': 64, 'occupancy': 4, 'group_size': 1}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 32.06 us | 42.37 % | 42.37 % | 47.52 % | 35.42 % | 32.46 % | 3.25 Tbyte/s | 128 | 135 register/thread | 0 byte/block | 65.58 Kbyte/block | 3 block / 3 block |
| fp16 | cutile | 34.85 us | 38.42 % | 38.42 % | 26.65 % | 24.40 % | 25.99 % | 2.94 Tbyte/s | 256 | 64 register/thread | 49.43 Kbyte/block | 0 byte/block | 4 block / 4 block |
| bf16 | triton | 32.10 us | 41.86 % | 41.86 % | 47.16 % | 34.79 % | 32.48 % | 3.21 Tbyte/s | 128 | 135 register/thread | 0 byte/block | 65.58 Kbyte/block | 3 block / 3 block |
| bf16 | cutile | 34.75 us | 37.93 % | 37.93 % | 25.89 % | 23.71 % | 26.26 % | 2.91 Tbyte/s | 256 | 64 register/thread | 49.43 Kbyte/block | 0 byte/block | 4 block / 4 block |
| fp32 | triton | 57.38 us | 41.77 % | 41.77 % | 30.55 % | 30.48 % | 32.24 % | 3.20 Tbyte/s | 128 | 135 register/thread | 0 byte/block | 98.35 Kbyte/block | 3 block / 2 block |
| fp32 | cutile | 99.81 us | 50.56 % | 24.60 % | 58.53 % | 16.72 % | 16.68 % | 1.89 Tbyte/s | 256 | 255 register/thread | 180.43 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.09× faster** (32.1 µs vs 34.9 µs).
- **bf16**: Triton is **1.08× faster** (32.1 µs vs 34.8 µs).
- **fp32**: Triton is **1.74× faster** (57.4 µs vs 99.8 µs).

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
