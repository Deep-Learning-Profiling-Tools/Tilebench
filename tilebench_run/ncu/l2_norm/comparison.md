# NCU Comparison: l2_norm

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'M': 2048, 'eps': 1e-06, 'K': 10240}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 2}` | `{'tile_size': 1024, 'occupancy': 16}` |
| bf16 | `{'batch': 1, 'M': 2048, 'eps': 1e-06, 'K': 10240}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 4}` | `{'tile_size': 1024, 'occupancy': 16}` |
| fp32 | `{'batch': 1, 'M': 2048, 'eps': 1e-06, 'K': 10240}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 4}` | `{'tile_size': 512, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 14.56 us | 39.89 % | 39.89 % | 56.75 % | 34.56 % | 25.12 % | 3.05 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp16 | cutile | 19.23 us | 30.46 % | 30.46 % | 40.92 % | 27.88 % | 35.29 % | 2.33 Tbyte/s | 128 | 29 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| bf16 | triton | 14.40 us | 40.46 % | 40.46 % | 56.39 % | 35.09 % | 27.26 % | 3.09 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| bf16 | cutile | 18.88 us | 30.87 % | 30.87 % | 39.95 % | 28.23 % | 36.97 % | 2.36 Tbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 27.49 us | 57.76 % | 57.76 % | 50.88 % | 38.70 % | 10.37 % | 4.42 Tbyte/s | 256 | 36 register/thread | 0 byte/block | 32 byte/block | 6 block / 14 block |
| fp32 | cutile | 30.40 us | 51.22 % | 51.22 % | 44.52 % | 33.19 % | 25.95 % | 3.92 Tbyte/s | 128 | 90 register/thread | 28 byte/block | 0 byte/block | 5 block / 14 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.32× faster** (14.6 µs vs 19.2 µs).
- **bf16**: Triton is **1.31× faster** (14.4 µs vs 18.9 µs).
- **fp32**: Triton is **1.11× faster** (27.5 µs vs 30.4 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **bf16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
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
