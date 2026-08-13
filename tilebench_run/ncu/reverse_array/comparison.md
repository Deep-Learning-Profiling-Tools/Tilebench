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
| fp16 | triton | 16.35 us | 68.84 % | 33.51 % | 91.99 % | 27.21 % | 27.99 % | 2.56 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 15.17 us | 42.70 % | 36.11 % | 60.33 % | 33.81 % | 28.22 % | 2.77 Tbyte/s | 128 | 88 register/thread | 16.40 Kbyte/block | 0 byte/block | 5 block / 7 block |
| bf16 | triton | 16.90 us | 66.84 % | 32.32 % | 89.28 % | 26.32 % | 27.10 % | 2.47 Tbyte/s | 256 | 40 register/thread | 0 byte/block | 0 byte/block | 6 block / 32 block |
| bf16 | cutile | 16.96 us | 46.57 % | 32.50 % | 61.46 % | 29.69 % | 36.02 % | 2.49 Tbyte/s | 128 | 23 register/thread | 2.06 Kbyte/block | 0 byte/block | 21 block / 32 block |
| fp32 | triton | 21.89 us | 64.56 % | 64.56 % | 78.86 % | 39.76 % | 20.09 % | 4.94 Tbyte/s | 128 | 29 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | cutile | 21.98 us | 65.12 % | 65.12 % | 63.42 % | 40.00 % | 34.13 % | 4.99 Tbyte/s | 128 | 28 register/thread | 4.11 Kbyte/block | 0 byte/block | 16 block / 25 block |
| int8 | triton | 14.46 us | 39.95 % | 18.07 % | 56.22 % | 15.60 % | 41.78 % | 1.38 Tbyte/s | 128 | 22 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| int8 | cutile | 12.61 us | 31.75 % | 20.73 % | 49.32 % | 24.90 % | 43.09 % | 1.59 Tbyte/s | 128 | 76 register/thread | 8.20 Kbyte/block | 0 byte/block | 6 block / 14 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.08× faster** (15.2 µs vs 16.4 µs).
- **bf16**: Triton is **1.00× faster** (16.9 µs vs 17.0 µs).
- **fp32**: Triton is **1.00× faster** (21.9 µs vs 22.0 µs).
- **int8**: cuTile is **1.15× faster** (12.6 µs vs 14.5 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — Memory is more heavily utilized than Compute
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
