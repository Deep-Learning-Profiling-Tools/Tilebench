# NCU Comparison: l2_norm

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'M': 2048, 'K': 10240, 'eps': 1e-06}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 2}` | `{'tile_size': 1024, 'occupancy': 16}` |
| bf16 | `{'batch': 1, 'M': 2048, 'K': 10240, 'eps': 1e-06}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 2}` | `{'tile_size': 2048, 'occupancy': 16}` |
| fp32 | `{'batch': 1, 'M': 2048, 'K': 10240, 'eps': 1e-06}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 3}` | `{'tile_size': 512, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 14.24 us | 40.61 % | 40.61 % | 55.93 % | 35.33 % | 24.88 % | 3.11 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp16 | cutile | 18.69 us | 31.37 % | 31.37 % | 40.33 % | 28.76 % | 35.85 % | 2.40 Tbyte/s | 128 | 29 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| bf16 | triton | 14.11 us | 41.18 % | 41.18 % | 56.57 % | 35.71 % | 26.71 % | 3.15 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| bf16 | cutile | 18.11 us | 32.28 % | 32.28 % | 41.62 % | 29.62 % | 30.23 % | 2.47 Tbyte/s | 128 | 32 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 27.62 us | 57.58 % | 57.58 % | 50.89 % | 38.43 % | 10.48 % | 4.41 Tbyte/s | 256 | 36 register/thread | 0 byte/block | 32 byte/block | 6 block / 14 block |
| fp32 | cutile | 30.56 us | 50.96 % | 50.96 % | 43.70 % | 32.99 % | 25.75 % | 3.90 Tbyte/s | 128 | 90 register/thread | 28 byte/block | 0 byte/block | 5 block / 14 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.31× faster** (14.2 µs vs 18.7 µs).
- **bf16**: Triton is **1.28× faster** (14.1 µs vs 18.1 µs).
- **fp32**: Triton is **1.11× faster** (27.6 µs vs 30.6 µs).

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
