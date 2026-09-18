# NCU Comparison: moe_topk_gating

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'E': 128, 'k': 2, 'M': 20480}` | `{'num_warps': 1}` | `{'occupancy': 8}` |
| bf16 | `{'E': 128, 'k': 2, 'M': 20480}` | `{'num_warps': 1}` | `{'occupancy': 32}` |
| fp32 | `{'E': 128, 'k': 2, 'M': 20480}` | `{'num_warps': 1}` | `{'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 15.71 us | 12.18 % | 5.84 % | 17.94 % | 3.09 % | 23.96 % | 446.71 Gbyte/s | 32 | 19 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| fp16 | cutile | 30.72 us | 63.45 % | 2.96 % | 72.13 % | 2.36 % | 67.15 % | 227.09 Gbyte/s | 128 | 22 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |
| bf16 | triton | 16.10 us | 12.21 % | 5.56 % | 17.48 % | 2.99 % | 24.46 % | 425.26 Gbyte/s | 32 | 19 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| bf16 | cutile | 30.40 us | 64.02 % | 2.90 % | 72.42 % | 2.13 % | 67.28 % | 222.09 Gbyte/s | 128 | 20 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |
| fp32 | triton | 15.68 us | 13.40 % | 10.93 % | 18.89 % | 5.80 % | 23.81 % | 837.84 Gbyte/s | 32 | 22 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| fp32 | cutile | 30.66 us | 63.59 % | 5.59 % | 72.54 % | 2.97 % | 66.78 % | 428.08 Gbyte/s | 128 | 22 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.96× faster** (15.7 µs vs 30.7 µs).
- **bf16**: Triton is **1.89× faster** (16.1 µs vs 30.4 µs).
- **fp32**: Triton is **1.96× faster** (15.7 µs vs 30.7 µs).

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
