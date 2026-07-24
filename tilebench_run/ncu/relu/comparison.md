# NCU Comparison: relu

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'tile': 2048, 'occupancy': 16}` |
| bf16 | `{'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'tile': 2048, 'occupancy': 16}` |
| fp32 | `{'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'tile': 2048, 'occupancy': 16}` |
| int8 | `{'n': 20971520}` | `(default)` | `(default)` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 14.91 us | 39.00 % | 39.00 % | 51.63 % | 31.09 % | 22.15 % | 2.98 Tbyte/s | 64 | 26 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| fp16 | cutile | 12.99 us | 44.58 % | 44.58 % | 56.40 % | 35.56 % | 36.84 % | 3.40 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| bf16 | triton | 14.85 us | 38.76 % | 38.76 % | 50.97 % | 31.19 % | 21.91 % | 2.97 Tbyte/s | 64 | 26 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | cutile | 12.96 us | 44.65 % | 44.65 % | 56.63 % | 35.79 % | 37.06 % | 3.42 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| fp32 | triton | 22.30 us | 66.72 % | 66.72 % | 56.80 % | 40.87 % | 14.60 % | 5.11 Tbyte/s | 256 | 18 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 22.43 us | 66.32 % | 66.32 % | 57.15 % | 40.30 % | 25.91 % | 5.08 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| int8 | triton | 14.85 us | 18.45 % | 18.45 % | 23.28 % | 15.83 % | 44.26 % | 1.41 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| int8 | cutile | 16.99 us | 16.13 % | 16.13 % | 19.55 % | 13.83 % | 52.37 % | 1.23 Tbyte/s | 128 | 17 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.15× faster** (13.0 µs vs 14.9 µs).
- **bf16**: cuTile is **1.15× faster** (13.0 µs vs 14.8 µs).
- **fp32**: Triton is **1.01× faster** (22.3 µs vs 22.4 µs).
- **int8**: Triton is **1.14× faster** (14.8 µs vs 17.0 µs).

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
