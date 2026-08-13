# NCU Comparison: matrix_copy

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'N': 5120}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 1024, 'occupancy': 8}` |
| bf16 | `{'N': 5120}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 2}` | `{'tile': 1024, 'occupancy': 16}` |
| fp32 | `{'N': 5120}` | `{'BLOCK_SIZE': 4096, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 1024, 'occupancy': 16}` |
| int8 | `{'N': 5120}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4, 'num_stages': 2}` | `{'tile': 2048, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 17.82 us | 46.37 % | 46.37 % | 45.43 % | 32.35 % | 16.18 % | 3.54 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| fp16 | cutile | 18.43 us | 44.64 % | 44.64 % | 45.15 % | 31.29 % | 28.97 % | 3.42 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | triton | 17.98 us | 45.74 % | 45.74 % | 45.44 % | 32.11 % | 15.37 % | 3.50 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | cutile | 18.34 us | 45.11 % | 45.11 % | 45.08 % | 31.40 % | 28.90 % | 3.45 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| fp32 | triton | 29.38 us | 68.70 % | 68.70 % | 53.36 % | 38.84 % | 9.89 % | 5.26 Tbyte/s | 128 | 45 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 27.94 us | 71.49 % | 71.49 % | 54.75 % | 40.83 % | 26.98 % | 5.48 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| int8 | triton | 10.82 us | 31.79 % | 31.79 % | 43.89 % | 27.07 % | 9.05 % | 2.43 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| int8 | cutile | 11.30 us | 30.44 % | 30.44 % | 41.87 % | 26.09 % | 23.03 % | 2.32 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.03× faster** (17.8 µs vs 18.4 µs).
- **bf16**: Triton is **1.02× faster** (18.0 µs vs 18.3 µs).
- **fp32**: cuTile is **1.05× faster** (27.9 µs vs 29.4 µs).
- **int8**: Triton is **1.04× faster** (10.8 µs vs 11.3 µs).

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
