# NCU Comparison: mul2

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'tile': 1024, 'occupancy': 4}` |
| bf16 | `{'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 2}` | `{'tile': 1024, 'occupancy': 32}` |
| fp32 | `{'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'tile': 2048, 'occupancy': 8}` |
| int8 | `{'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 2}` | `{'tile': 2048, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 14.72 us | 39.24 % | 39.24 % | 51.98 % | 31.15 % | 7.73 % | 3.01 Tbyte/s | 64 | 18 register/thread | 0 byte/block | 0 byte/block | 42 block / 32 block |
| fp16 | cutile | 15.68 us | 37.15 % | 37.15 % | 45.32 % | 29.63 % | 31.88 % | 2.85 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | triton | 13.54 us | 42.53 % | 42.53 % | 58.18 % | 33.26 % | 8.12 % | 3.26 Tbyte/s | 64 | 26 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | cutile | 15.36 us | 37.95 % | 37.95 % | 45.41 % | 30.18 % | 31.00 % | 2.90 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| fp32 | triton | 22.40 us | 66.43 % | 66.43 % | 56.49 % | 40.77 % | 10.52 % | 5.08 Tbyte/s | 256 | 18 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 22.66 us | 65.05 % | 65.05 % | 56.78 % | 39.93 % | 23.52 % | 4.99 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| int8 | triton | 9.73 us | 28.20 % | 28.20 % | 45.45 % | 24.08 % | 36.39 % | 2.16 Tbyte/s | 64 | 29 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| int8 | cutile | 11.23 us | 24.47 % | 24.47 % | 34.52 % | 21.02 % | 38.42 % | 1.87 Tbyte/s | 128 | 20 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.07× faster** (14.7 µs vs 15.7 µs).
- **bf16**: Triton is **1.13× faster** (13.5 µs vs 15.4 µs).
- **fp32**: Triton is **1.01× faster** (22.4 µs vs 22.7 µs).
- **int8**: Triton is **1.15× faster** (9.7 µs vs 11.2 µs).

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
