# NCU Comparison: cross_entropy

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'num_classes': 512, 'batch_size': 10240}` | `{'num_warps': 1}` | `{'occupancy': 8}` |
| fp32 | `{'num_classes': 512, 'batch_size': 10240}` | `{'num_warps': 1}` | `{'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 9.82 us | 14.12 % | 14.12 % | 11.74 % | 9.40 % | 20.05 % | 1.08 Tbyte/s | 32 | 26 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| fp16 | cutile | 14.69 us | 32.71 % | 9.42 % | 46.61 % | 6.25 % | 54.78 % | 719.90 Gbyte/s | 128 | 22 register/thread | 20 byte/block | 0 byte/block | 21 block / 28 block |
| fp32 | triton | 9.95 us | 27.66 % | 27.66 % | 16.13 % | 18.48 % | 18.39 % | 2.12 Tbyte/s | 32 | 26 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| fp32 | cutile | 13.82 us | 31.90 % | 19.89 % | 44.63 % | 13.24 % | 50.12 % | 1.52 Tbyte/s | 128 | 21 register/thread | 28 byte/block | 0 byte/block | 21 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.50× faster** (9.8 µs vs 14.7 µs).
- **fp32**: Triton is **1.39× faster** (9.9 µs vs 13.8 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
