# NCU Comparison: l2_norm

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'M': 2048, 'K': 10240, 'eps': 1e-06}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 3}` | `{'tile_size': 512, 'occupancy': 4}` |
| bf16 | `{'batch': 1, 'M': 2048, 'K': 10240, 'eps': 1e-06}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 2}` | `{'tile_size': 1024, 'occupancy': 16}` |
| fp32 | `{'batch': 1, 'M': 2048, 'K': 10240, 'eps': 1e-06}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 3}` | `{'tile_size': 512, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 14.30 us | 40.45 % | 40.45 % | 56.36 % | 35.31 % | 25.15 % | 3.10 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp16 | cutile | 34.08 us | 16.96 % | 16.96 % | 17.77 % | 13.99 % | 28.95 % | 1.30 Tbyte/s | 128 | 82 register/thread | 28 byte/block | 0 byte/block | 5 block / 14 block |
| bf16 | triton | 14.24 us | 40.80 % | 40.80 % | 56.42 % | 35.10 % | 26.43 % | 3.13 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| bf16 | cutile | 18.85 us | 31.02 % | 31.02 % | 40.85 % | 28.51 % | 34.94 % | 2.37 Tbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 27.39 us | 57.89 % | 57.89 % | 51.19 % | 38.64 % | 10.28 % | 4.44 Tbyte/s | 256 | 36 register/thread | 0 byte/block | 32 byte/block | 6 block / 14 block |
| fp32 | cutile | 30.02 us | 51.80 % | 51.80 % | 44.19 % | 33.68 % | 25.91 % | 3.97 Tbyte/s | 128 | 90 register/thread | 28 byte/block | 0 byte/block | 5 block / 14 block |

## Key findings (auto-derived)

- **fp16**: Triton is **2.38× faster** (14.3 µs vs 34.1 µs).
- **bf16**: Triton is **1.32× faster** (14.2 µs vs 18.9 µs).
- **fp32**: Triton is **1.10× faster** (27.4 µs vs 30.0 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **bf16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
