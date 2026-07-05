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
| fp16 | triton | 14.85 us | 38.97 % | 38.97 % | 49.98 % | 31.17 % | 7.74 % | 2.98 Tbyte/s | 64 | 18 register/thread | 0 byte/block | 0 byte/block | 42 block / 32 block |
| fp16 | cutile | 15.20 us | 38.30 % | 38.30 % | 45.00 % | 30.64 % | 31.53 % | 2.93 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | triton | 13.44 us | 42.98 % | 42.98 % | 58.08 % | 33.70 % | 8.36 % | 3.28 Tbyte/s | 64 | 26 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | cutile | 15.26 us | 38.13 % | 38.13 % | 45.38 % | 30.44 % | 31.49 % | 2.92 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| fp32 | triton | 22.11 us | 66.96 % | 66.96 % | 57.30 % | 41.28 % | 10.11 % | 5.13 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| fp32 | cutile | 22.11 us | 66.76 % | 66.76 % | 56.57 % | 41.38 % | 29.40 % | 5.12 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| int8 | triton | 9.73 us | 28.24 % | 28.24 % | 47.85 % | 24.12 % | 34.27 % | 2.16 Tbyte/s | 64 | 29 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| int8 | cutile | 15.14 us | 18.15 % | 18.15 % | 22.46 % | 15.54 % | 41.08 % | 1.39 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.02× faster** (14.8 µs vs 15.2 µs).
- **bf16**: Triton is **1.14× faster** (13.4 µs vs 15.3 µs).
- **fp32**: cuTile is **1.00× faster** (22.1 µs vs 22.1 µs).
- **int8**: Triton is **1.56× faster** (9.7 µs vs 15.1 µs).

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
