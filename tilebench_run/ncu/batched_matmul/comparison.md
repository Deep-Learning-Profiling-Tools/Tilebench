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
| fp16 | triton | 29.73 us | 41.49 % | 23.99 % | 51.48 % | 27.40 % | 37.85 % | 1.84 Tbyte/s | 128 | 149 register/thread | 0 byte/block | 65.55 Kbyte/block | 3 block / 3 block |
| fp16 | cutile | 30.62 us | 27.13 % | 23.41 % | 32.67 % | 24.46 % | 30.90 % | 1.79 Tbyte/s | 256 | 64 register/thread | 49.43 Kbyte/block | 0 byte/block | 4 block / 4 block |
| bf16 | triton | 29.60 us | 40.74 % | 24.08 % | 51.69 % | 27.35 % | 37.27 % | 1.84 Tbyte/s | 128 | 149 register/thread | 0 byte/block | 65.55 Kbyte/block | 3 block / 3 block |
| bf16 | cutile | 30.24 us | 27.13 % | 23.53 % | 32.06 % | 25.40 % | 30.79 % | 1.80 Tbyte/s | 256 | 64 register/thread | 49.43 Kbyte/block | 0 byte/block | 4 block / 4 block |
| fp32 | triton | 117.86 us | 69.02 % | 14.71 % | 73.13 % | 40.69 % | 55.60 % | 1.13 Tbyte/s | 128 | 96 register/thread | 0 byte/block | 40.96 Kbyte/block | 5 block / 5 block |
| fp32 | cutile | 98.53 us | 50.61 % | 17.24 % | 59.34 % | 15.01 % | 16.69 % | 1.32 Tbyte/s | 256 | 255 register/thread | 180.43 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.03× faster** (29.7 µs vs 30.6 µs).
- **bf16**: Triton is **1.02× faster** (29.6 µs vs 30.2 µs).
- **fp32**: cuTile is **1.20× faster** (98.5 µs vs 117.9 µs).

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
