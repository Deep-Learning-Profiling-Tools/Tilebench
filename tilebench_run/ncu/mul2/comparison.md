# NCU Comparison: mul2

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'tile': 1024, 'occupancy': 16}` |
| bf16 | `{'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 2}` | `{'tile': 1024, 'occupancy': 16}` |
| fp32 | `{'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4}` | `{'tile': 1024, 'occupancy': 16}` |
| int8 | `{'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 2}` | `{'tile': 1024, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 14.82 us | 39.06 % | 39.06 % | 51.05 % | 31.25 % | 7.60 % | 2.99 Tbyte/s | 64 | 18 register/thread | 0 byte/block | 0 byte/block | 42 block / 32 block |
| fp16 | cutile | 15.33 us | 37.96 % | 37.96 % | 44.94 % | 30.26 % | 31.17 % | 2.90 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | triton | 13.57 us | 42.40 % | 42.40 % | 55.98 % | 33.09 % | 8.28 % | 3.25 Tbyte/s | 64 | 26 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | cutile | 15.33 us | 38.08 % | 38.08 % | 44.69 % | 30.43 % | 31.19 % | 2.91 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| fp32 | triton | 22.02 us | 67.54 % | 67.54 % | 56.40 % | 41.32 % | 10.50 % | 5.17 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| fp32 | cutile | 22.40 us | 66.51 % | 66.51 % | 56.32 % | 40.85 % | 29.79 % | 5.10 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| int8 | triton | 9.92 us | 27.70 % | 27.70 % | 47.59 % | 23.50 % | 36.23 % | 2.11 Tbyte/s | 64 | 29 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| int8 | cutile | 15.33 us | 17.93 % | 17.93 % | 22.57 % | 15.41 % | 42.07 % | 1.37 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.03× faster** (14.8 µs vs 15.3 µs).
- **bf16**: Triton is **1.13× faster** (13.6 µs vs 15.3 µs).
- **fp32**: Triton is **1.02× faster** (22.0 µs vs 22.4 µs).
- **int8**: Triton is **1.55× faster** (9.9 µs vs 15.3 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **bf16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — Memory is more heavily utilized than Compute
- **fp32 / triton** — Memory is more heavily utilized than Compute
- **int8 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **int8 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `cutile_int8.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`
- `triton_int8.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
