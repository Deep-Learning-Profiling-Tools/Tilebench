# NCU Comparison: reverse_array

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 20000000}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4, 'num_stages': 2}` | `{'tile': 2048, 'occupancy': 4}` |
| bf16 | `{'n': 20000000}` | `{'BLOCK_SIZE': 8192, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 2048, 'occupancy': 4}` |
| fp32 | `{'n': 20000000}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 2048, 'occupancy': 4}` |
| int8 | `{'n': 20000000}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 2048, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 17.02 us | 65.73 % | 31.96 % | 93.48 % | 25.99 % | 26.71 % | 2.44 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 14.37 us | 46.69 % | 37.96 % | 67.29 % | 35.89 % | 40.26 % | 2.91 Tbyte/s | 128 | 30 register/thread | 4.11 Kbyte/block | 0 byte/block | 16 block / 25 block |
| bf16 | triton | 16.70 us | 65.97 % | 32.55 % | 89.86 % | 26.59 % | 26.62 % | 2.49 Tbyte/s | 256 | 40 register/thread | 0 byte/block | 0 byte/block | 6 block / 32 block |
| bf16 | cutile | 14.72 us | 44.90 % | 37.02 % | 67.25 % | 35.18 % | 38.73 % | 2.83 Tbyte/s | 128 | 30 register/thread | 4.11 Kbyte/block | 0 byte/block | 16 block / 25 block |
| fp32 | triton | 21.54 us | 65.71 % | 65.71 % | 78.75 % | 40.40 % | 20.02 % | 5.03 Tbyte/s | 128 | 29 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | cutile | 24.48 us | 59.36 % | 59.36 % | 53.31 % | 35.67 % | 21.08 % | 4.55 Tbyte/s | 128 | 38 register/thread | 8.20 Kbyte/block | 0 byte/block | 12 block / 14 block |
| int8 | triton | 14.40 us | 39.97 % | 18.18 % | 55.14 % | 15.71 % | 41.86 % | 1.39 Tbyte/s | 128 | 22 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| int8 | cutile | 13.57 us | 40.07 % | 19.28 % | 60.70 % | 23.53 % | 32.23 % | 1.47 Tbyte/s | 128 | 31 register/thread | 2.06 Kbyte/block | 0 byte/block | 16 block / 32 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.18× faster** (14.4 µs vs 17.0 µs).
- **bf16**: cuTile is **1.13× faster** (14.7 µs vs 16.7 µs).
- **fp32**: Triton is **1.14× faster** (21.5 µs vs 24.5 µs).
- **int8**: cuTile is **1.06× faster** (13.6 µs vs 14.4 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
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
