# NCU Comparison: cross_entropy

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'num_classes': 512, 'batch_size': 10240}` | `{'num_warps': 1}` | `{'occupancy': 8}` |
| fp32 | `{'num_classes': 512, 'batch_size': 10240}` | `{'num_warps': 1}` | `{'occupancy': 32}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 9.82 us | 14.07 % | 14.07 % | 11.99 % | 9.38 % | 20.50 % | 1.08 Tbyte/s | 32 | 26 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| fp16 | cutile | 14.37 us | 27.07 % | 9.63 % | 40.54 % | 6.40 % | 47.11 % | 735.89 Gbyte/s | 128 | 21 register/thread | 28 byte/block | 0 byte/block | 21 block / 28 block |
| fp32 | triton | 10.08 us | 27.33 % | 27.33 % | 17.46 % | 18.33 % | 19.09 % | 2.09 Tbyte/s | 32 | 26 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| fp32 | cutile | 13.86 us | 32.30 % | 19.86 % | 44.21 % | 13.19 % | 50.86 % | 1.52 Tbyte/s | 128 | 24 register/thread | 28 byte/block | 0 byte/block | 21 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.46× faster** (9.8 µs vs 14.4 µs).
- **fp32**: Triton is **1.38× faster** (10.1 µs vs 13.9 µs).

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
