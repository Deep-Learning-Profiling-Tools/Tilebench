# NCU Comparison: moe_topk_gating

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'E': 128, 'k': 2, 'M': 20480}` | `{'num_warps': 1}` | `{'occupancy': 16}` |
| bf16 | `{'E': 128, 'k': 2, 'M': 20480}` | `{'num_warps': 1}` | `{'occupancy': 32}` |
| fp32 | `{'E': 128, 'k': 2, 'M': 20480}` | `{'num_warps': 1}` | `{'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 14.78 us | 13.06 % | 4.64 % | 19.51 % | 3.08 % | 25.70 % | 355.00 Gbyte/s | 32 | 19 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| fp16 | cutile | 29.63 us | 64.87 % | 2.31 % | 74.59 % | 2.34 % | 68.66 % | 177.22 Gbyte/s | 128 | 22 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |
| bf16 | triton | 14.78 us | 12.35 % | 4.64 % | 19.47 % | 3.09 % | 24.74 % | 355.00 Gbyte/s | 32 | 19 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| bf16 | cutile | 29.28 us | 65.63 % | 2.34 % | 75.57 % | 2.11 % | 68.97 % | 179.35 Gbyte/s | 128 | 20 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |
| fp32 | triton | 14.88 us | 14.06 % | 9.24 % | 20.84 % | 6.14 % | 24.98 % | 705.03 Gbyte/s | 32 | 22 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| fp32 | cutile | 29.66 us | 65.61 % | 4.62 % | 74.85 % | 3.06 % | 68.91 % | 353.77 Gbyte/s | 128 | 22 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **2.00× faster** (14.8 µs vs 29.6 µs).
- **bf16**: Triton is **1.98× faster** (14.8 µs vs 29.3 µs).
- **fp32**: Triton is **1.99× faster** (14.9 µs vs 29.7 µs).

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
