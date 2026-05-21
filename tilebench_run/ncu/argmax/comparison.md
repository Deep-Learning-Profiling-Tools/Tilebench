# NCU Comparison: argmax

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_N': 2048, 'num_warps': 4, 'num_stages': 2}` | `{'block_n': 2048, 'occupancy': 4}` |
| fp32 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_N': 1024, 'num_warps': 4, 'num_stages': 2}` | `{'block_n': 2048, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 30.11 us | 37.19 % | 37.19 % | 44.25 % | 24.25 % | 59.60 % | 2.85 Tbyte/s | 128 | 38 register/thread | 0 byte/block | 32 byte/block | 12 block / 28 block |
| fp16 | cutile | 47.81 us | 27.50 % | 23.46 % | 31.27 % | 15.20 % | 45.90 % | 1.80 Tbyte/s | 128 | 62 register/thread | 44 byte/block | 0 byte/block | 8 block / 28 block |
| fp32 | triton | 40.74 us | 54.76 % | 54.76 % | 58.75 % | 35.87 % | 60.38 % | 4.19 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 32 byte/block | 16 block / 28 block |
| fp32 | cutile | 39.36 us | 56.74 % | 56.74 % | 40.71 % | 37.06 % | 53.79 % | 4.34 Tbyte/s | 128 | 60 register/thread | 44 byte/block | 0 byte/block | 8 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.59× faster** (30.1 µs vs 47.8 µs).
- **fp32**: cuTile is **1.04× faster** (39.4 µs vs 40.7 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / triton** — Compute and Memory are well-balanced

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
