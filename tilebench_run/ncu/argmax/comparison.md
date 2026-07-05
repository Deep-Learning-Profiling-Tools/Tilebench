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
| fp16 | triton | 30.11 us | 37.28 % | 37.28 % | 44.24 % | 24.25 % | 58.28 % | 2.86 Tbyte/s | 128 | 38 register/thread | 0 byte/block | 32 byte/block | 12 block / 28 block |
| fp16 | cutile | 48.45 us | 27.83 % | 23.24 % | 31.41 % | 15.02 % | 46.45 % | 1.78 Tbyte/s | 128 | 62 register/thread | 44 byte/block | 0 byte/block | 8 block / 28 block |
| fp32 | triton | 41.09 us | 54.24 % | 54.24 % | 58.68 % | 35.54 % | 60.49 % | 4.16 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 32 byte/block | 16 block / 28 block |
| fp32 | cutile | 38.46 us | 57.96 % | 57.96 % | 41.09 % | 37.95 % | 53.23 % | 4.44 Tbyte/s | 128 | 60 register/thread | 44 byte/block | 0 byte/block | 8 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.61× faster** (30.1 µs vs 48.5 µs).
- **fp32**: cuTile is **1.07× faster** (38.5 µs vs 41.1 µs).

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
