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
| fp16 | triton | 16.48 us | 68.03 % | 33.44 % | 91.81 % | 27.02 % | 27.61 % | 2.56 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 14.14 us | 46.77 % | 39.07 % | 66.75 % | 36.69 % | 40.30 % | 2.98 Tbyte/s | 128 | 30 register/thread | 4.11 Kbyte/block | 0 byte/block | 16 block / 25 block |
| bf16 | triton | 16.58 us | 66.66 % | 32.95 % | 89.46 % | 26.97 % | 26.91 % | 2.51 Tbyte/s | 256 | 40 register/thread | 0 byte/block | 0 byte/block | 6 block / 32 block |
| bf16 | cutile | 14.27 us | 44.33 % | 38.93 % | 64.04 % | 36.37 % | 38.28 % | 2.97 Tbyte/s | 128 | 30 register/thread | 4.11 Kbyte/block | 0 byte/block | 16 block / 25 block |
| fp32 | triton | 21.47 us | 66.39 % | 66.39 % | 77.98 % | 40.65 % | 20.05 % | 5.09 Tbyte/s | 128 | 29 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | cutile | 24.99 us | 58.85 % | 58.85 % | 53.91 % | 35.11 % | 21.88 % | 4.50 Tbyte/s | 128 | 38 register/thread | 8.20 Kbyte/block | 0 byte/block | 12 block / 14 block |
| int8 | triton | 14.56 us | 40.56 % | 17.95 % | 53.93 % | 15.54 % | 42.58 % | 1.37 Tbyte/s | 128 | 22 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| int8 | cutile | 13.76 us | 43.25 % | 19.03 % | 61.70 % | 23.22 % | 34.87 % | 1.45 Tbyte/s | 128 | 31 register/thread | 2.06 Kbyte/block | 0 byte/block | 16 block / 32 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.17× faster** (14.1 µs vs 16.5 µs).
- **bf16**: cuTile is **1.16× faster** (14.3 µs vs 16.6 µs).
- **fp32**: Triton is **1.16× faster** (21.5 µs vs 25.0 µs).
- **int8**: cuTile is **1.06× faster** (13.8 µs vs 14.6 µs).

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
