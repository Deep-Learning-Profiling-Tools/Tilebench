# NCU Comparison: moe_topk_gating

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'E': 128, 'k': 2, 'M': 20480}` | `{'num_warps': 1}` | `{'occupancy': 32}` |
| bf16 | `{'E': 128, 'k': 2, 'M': 20480}` | `{'num_warps': 1}` | `{'occupancy': 32}` |
| fp32 | `{'E': 128, 'k': 2, 'M': 20480}` | `{'num_warps': 1}` | `{'occupancy': 32}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 15.01 us | 12.64 % | 4.58 % | 19.21 % | 3.04 % | 27.66 % | 349.70 Gbyte/s | 32 | 22 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| fp16 | cutile | 29.09 us | 64.79 % | 2.36 % | 75.56 % | 2.12 % | 68.09 % | 180.53 Gbyte/s | 128 | 20 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |
| bf16 | triton | 14.59 us | 12.65 % | 4.71 % | 18.95 % | 3.13 % | 25.35 % | 359.67 Gbyte/s | 32 | 20 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| bf16 | cutile | 29.12 us | 65.15 % | 2.36 % | 75.58 % | 2.11 % | 68.46 % | 180.33 Gbyte/s | 128 | 20 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |
| fp32 | triton | 14.59 us | 13.66 % | 9.40 % | 20.92 % | 6.24 % | 24.27 % | 718.93 Gbyte/s | 32 | 22 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| fp32 | cutile | 29.09 us | 65.16 % | 4.71 % | 74.89 % | 3.13 % | 67.96 % | 360.77 Gbyte/s | 128 | 20 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.94× faster** (15.0 µs vs 29.1 µs).
- **bf16**: Triton is **2.00× faster** (14.6 µs vs 29.1 µs).
- **fp32**: Triton is **1.99× faster** (14.6 µs vs 29.1 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Compute and Memory are well-balanced
- **bf16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / cutile** — Compute and Memory are well-balanced
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — Compute and Memory are well-balanced
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
