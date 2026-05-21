# NCU Comparison: matrix_copy

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'N': 5120}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 1024, 'occupancy': 8}` |
| bf16 | `{'N': 5120}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 1024, 'occupancy': 8}` |
| fp32 | `{'N': 5120}` | `{'BLOCK_SIZE': 1024, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 1024, 'occupancy': 8}` |
| int8 | `{'N': 5120}` | `{'BLOCK_SIZE': 4096, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 1024, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 17.76 us | 45.98 % | 45.98 % | 45.43 % | 32.47 % | 15.87 % | 3.51 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| fp16 | cutile | 18.05 us | 45.52 % | 45.52 % | 44.54 % | 32.02 % | 29.02 % | 3.49 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | triton | 18.02 us | 45.41 % | 45.41 % | 45.13 % | 32.00 % | 15.84 % | 3.48 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | cutile | 17.98 us | 45.61 % | 45.61 % | 44.93 % | 32.08 % | 29.31 % | 3.49 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| fp32 | triton | 31.23 us | 68.50 % | 68.50 % | 47.56 % | 37.09 % | 18.33 % | 5.25 Tbyte/s | 256 | 16 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | cutile | 27.97 us | 72.01 % | 72.01 % | 54.34 % | 40.72 % | 25.34 % | 5.52 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| int8 | triton | 9.79 us | 34.99 % | 34.99 % | 52.79 % | 30.20 % | 10.41 % | 2.68 Tbyte/s | 256 | 16 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| int8 | cutile | 17.73 us | 19.33 % | 19.33 % | 23.28 % | 16.52 % | 30.39 % | 1.48 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.02× faster** (17.8 µs vs 18.1 µs).
- **bf16**: cuTile is **1.00× faster** (18.0 µs vs 18.0 µs).
- **fp32**: cuTile is **1.12× faster** (28.0 µs vs 31.2 µs).
- **int8**: Triton is **1.81× faster** (9.8 µs vs 17.7 µs).

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
