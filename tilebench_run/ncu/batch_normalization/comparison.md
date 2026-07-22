# NCU Comparison: batch_normalization

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'N': 20000, 'C': 1024, 'eps': 1e-05}` | `{'ROWS': 8, 'num_warps': 8}` | `{'rows': 16, 'occupancy': 8}` |
| bf16 | `{'N': 20000, 'C': 1024, 'eps': 1e-05}` | `{'ROWS': 8, 'num_warps': 8}` | `{'rows': 16, 'occupancy': 8}` |
| fp32 | `{'N': 20000, 'C': 1024, 'eps': 1e-05}` | `{'ROWS': 8, 'num_warps': 8}` | `{'rows': 8, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 33.18 us | 41.40 % | 40.32 % | 63.03 % | 32.74 % | 20.93 % | 3.08 Tbyte/s | 256 | 48 register/thread | 0 byte/block | 4.10 Kbyte/block | 5 block / 12 block |
| fp16 | cutile | 53.92 us | 20.42 % | 20.42 % | 28.54 % | 16.83 % | 19.19 % | 1.56 Tbyte/s | 128 | 62 register/thread | 16.40 Kbyte/block | 0 byte/block | 8 block / 9 block |
| bf16 | triton | 33.28 us | 38.62 % | 38.62 % | 54.94 % | 31.06 % | 21.88 % | 2.96 Tbyte/s | 256 | 48 register/thread | 0 byte/block | 4.10 Kbyte/block | 5 block / 12 block |
| bf16 | cutile | 53.99 us | 20.44 % | 20.44 % | 28.18 % | 16.92 % | 19.00 % | 1.56 Tbyte/s | 128 | 62 register/thread | 16.40 Kbyte/block | 0 byte/block | 8 block / 9 block |
| fp32 | triton | 60.67 us | 35.56 % | 35.56 % | 9.02 % | 23.23 % | 8.41 % | 2.73 Tbyte/s | 128 | 132 register/thread | 0 byte/block | 0 byte/block | 3 block / 32 block |
| fp32 | cutile | 52.80 us | 59.48 % | 59.48 % | 49.93 % | 35.98 % | 22.93 % | 4.56 Tbyte/s | 128 | 92 register/thread | 0 byte/block | 0 byte/block | 5 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.63× faster** (33.2 µs vs 53.9 µs).
- **bf16**: Triton is **1.62× faster** (33.3 µs vs 54.0 µs).
- **fp32**: cuTile is **1.15× faster** (52.8 µs vs 60.7 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **bf16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.70 full waves across all SMs. Look at Launch Statistics for more details.

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
