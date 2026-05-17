# NCU Comparison: moe_topk_gating

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'E': 128, 'k': 2, 'M': 20480}` | `{'num_warps': 1}` | `{'occupancy': 8}` |
| bf16 | `{'E': 128, 'k': 2, 'M': 20480}` | `{'num_warps': 1}` | `{'occupancy': 8}` |
| fp32 | `{'E': 128, 'k': 2, 'M': 20480}` | `{'num_warps': 1}` | `{'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 14.78 us | 12.83 % | 4.64 % | 18.44 % | 3.09 % | 31.63 % | 354.94 Gbyte/s | 32 | 23 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| fp16 | cutile | 29.47 us | 65.03 % | 2.33 % | 74.74 % | 2.36 % | 68.82 % | 178.19 Gbyte/s | 128 | 22 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |
| bf16 | triton | 14.66 us | 12.94 % | 4.67 % | 18.90 % | 3.10 % | 29.49 % | 358.06 Gbyte/s | 32 | 26 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| bf16 | cutile | 30.30 us | 64.80 % | 2.26 % | 74.64 % | 2.32 % | 68.58 % | 173.29 Gbyte/s | 128 | 22 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |
| fp32 | triton | 14.69 us | 13.43 % | 9.32 % | 20.15 % | 6.20 % | 26.86 % | 714.20 Gbyte/s | 32 | 28 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| fp32 | cutile | 29.38 us | 65.34 % | 4.66 % | 74.95 % | 3.10 % | 68.61 % | 357.24 Gbyte/s | 128 | 22 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.99× faster** (14.8 µs vs 29.5 µs).
- **bf16**: Triton is **2.07× faster** (14.7 µs vs 30.3 µs).
- **fp32**: Triton is **2.00× faster** (14.7 µs vs 29.4 µs).

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
