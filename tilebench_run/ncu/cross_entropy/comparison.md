# NCU Comparison: cross_entropy

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'num_classes': 512, 'batch_size': 10240}` | `{'num_warps': 1}` | `{'occupancy': 32}` |
| fp32 | `{'num_classes': 512, 'batch_size': 10240}` | `{'num_warps': 1}` | `{'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 9.86 us | 14.05 % | 14.05 % | 11.55 % | 9.35 % | 20.21 % | 1.07 Tbyte/s | 32 | 26 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| fp16 | cutile | 15.04 us | 30.92 % | 9.19 % | 43.54 % | 6.10 % | 52.36 % | 703.05 Gbyte/s | 128 | 21 register/thread | 20 byte/block | 0 byte/block | 21 block / 28 block |
| fp32 | triton | 10.46 us | 26.34 % | 26.34 % | 16.92 % | 17.62 % | 19.81 % | 2.01 Tbyte/s | 32 | 26 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| fp32 | cutile | 13.89 us | 30.31 % | 19.86 % | 42.55 % | 13.20 % | 47.51 % | 1.52 Tbyte/s | 128 | 21 register/thread | 28 byte/block | 0 byte/block | 21 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.53× faster** (9.9 µs vs 15.0 µs).
- **fp32**: Triton is **1.33× faster** (10.5 µs vs 13.9 µs).

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
