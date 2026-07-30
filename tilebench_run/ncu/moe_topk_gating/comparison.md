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
| fp16 | triton | 14.66 us | 12.62 % | 4.68 % | 19.10 % | 3.11 % | 27.61 % | 358.10 Gbyte/s | 32 | 22 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| fp16 | cutile | 29.09 us | 64.99 % | 2.36 % | 75.61 % | 2.12 % | 68.30 % | 180.53 Gbyte/s | 128 | 20 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |
| bf16 | triton | 14.59 us | 12.65 % | 4.70 % | 19.57 % | 3.12 % | 25.37 % | 359.67 Gbyte/s | 32 | 20 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| bf16 | cutile | 29.06 us | 65.28 % | 2.36 % | 75.56 % | 2.11 % | 68.60 % | 180.74 Gbyte/s | 128 | 20 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |
| fp32 | triton | 14.69 us | 13.69 % | 9.32 % | 20.67 % | 6.19 % | 24.33 % | 714.23 Gbyte/s | 32 | 22 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| fp32 | cutile | 29.22 us | 65.06 % | 4.69 % | 75.71 % | 3.11 % | 67.85 % | 359.20 Gbyte/s | 128 | 20 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.98× faster** (14.7 µs vs 29.1 µs).
- **bf16**: Triton is **1.99× faster** (14.6 µs vs 29.1 µs).
- **fp32**: Triton is **1.99× faster** (14.7 µs vs 29.2 µs).

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
