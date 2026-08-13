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
| fp16 | triton | 32.32 us | 41.72 % | 41.72 % | 60.45 % | 33.66 % | 20.34 % | 3.19 Tbyte/s | 256 | 48 register/thread | 0 byte/block | 4.10 Kbyte/block | 5 block / 12 block |
| fp16 | cutile | 54.69 us | 19.90 % | 19.90 % | 27.81 % | 16.37 % | 19.37 % | 1.52 Tbyte/s | 128 | 62 register/thread | 16.40 Kbyte/block | 0 byte/block | 8 block / 9 block |
| bf16 | triton | 33.15 us | 39.20 % | 39.20 % | 56.02 % | 31.51 % | 22.64 % | 3.00 Tbyte/s | 256 | 48 register/thread | 0 byte/block | 4.10 Kbyte/block | 5 block / 12 block |
| bf16 | cutile | 54.98 us | 19.66 % | 19.66 % | 26.24 % | 16.19 % | 19.04 % | 1.50 Tbyte/s | 128 | 62 register/thread | 16.40 Kbyte/block | 0 byte/block | 8 block / 9 block |
| fp32 | triton | 61.00 us | 35.30 % | 35.30 % | 9.03 % | 22.90 % | 8.46 % | 2.70 Tbyte/s | 128 | 132 register/thread | 0 byte/block | 0 byte/block | 3 block / 32 block |
| fp32 | cutile | 52.18 us | 59.74 % | 59.74 % | 51.09 % | 36.21 % | 23.31 % | 4.57 Tbyte/s | 128 | 92 register/thread | 0 byte/block | 0 byte/block | 5 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.69× faster** (32.3 µs vs 54.7 µs).
- **bf16**: Triton is **1.66× faster** (33.1 µs vs 55.0 µs).
- **fp32**: cuTile is **1.17× faster** (52.2 µs vs 61.0 µs).

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
