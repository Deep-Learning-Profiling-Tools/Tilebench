# NCU Comparison: reverse_array

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 20000000}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 8192, 'occupancy': 16}` |
| bf16 | `{'n': 20000000}` | `{'BLOCK_SIZE': 8192, 'num_warps': 8, 'num_stages': 2}` | `{'tile': 1024, 'occupancy': 16}` |
| fp32 | `{'n': 20000000}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 2}` | `{'tile': 1024, 'occupancy': 8}` |
| int8 | `{'n': 20000000}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 8192, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 19.58 us | 56.46 % | 55.05 % | 74.50 % | 24.76 % | 23.21 % | 4.21 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 20.29 us | 54.01 % | 54.01 % | 38.31 % | 25.37 % | 20.62 % | 4.13 Tbyte/s | 128 | 88 register/thread | 16.40 Kbyte/block | 0 byte/block | 5 block / 7 block |
| bf16 | triton | 19.55 us | 58.88 % | 54.29 % | 77.13 % | 24.76 % | 23.98 % | 4.15 Tbyte/s | 256 | 40 register/thread | 0 byte/block | 0 byte/block | 6 block / 32 block |
| bf16 | cutile | 20.93 us | 52.77 % | 52.77 % | 44.57 % | 23.77 % | 27.38 % | 4.04 Tbyte/s | 128 | 23 register/thread | 2.06 Kbyte/block | 0 byte/block | 21 block / 32 block |
| fp32 | triton | 28.38 us | 74.00 % | 74.00 % | 54.26 % | 33.96 % | 15.27 % | 5.67 Tbyte/s | 128 | 29 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | cutile | 29.92 us | 70.05 % | 70.05 % | 43.49 % | 32.11 % | 26.19 % | 5.37 Tbyte/s | 128 | 28 register/thread | 4.11 Kbyte/block | 0 byte/block | 16 block / 25 block |
| int8 | triton | 15.07 us | 38.30 % | 33.91 % | 51.96 % | 15.77 % | 39.96 % | 2.60 Tbyte/s | 128 | 22 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| int8 | cutile | 15.07 us | 33.40 % | 33.40 % | 37.03 % | 20.84 % | 37.80 % | 2.56 Tbyte/s | 128 | 76 register/thread | 8.20 Kbyte/block | 0 byte/block | 6 block / 14 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.04× faster** (19.6 µs vs 20.3 µs).
- **bf16**: Triton is **1.07× faster** (19.6 µs vs 20.9 µs).
- **fp32**: Triton is **1.05× faster** (28.4 µs vs 29.9 µs).
- **int8**: cuTile is **1.00× faster** (15.1 µs vs 15.1 µs).

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
