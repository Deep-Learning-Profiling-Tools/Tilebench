# NCU Comparison: layernorm

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'M': 2048, 'K': 10240}` | `{'BLOCK_N_SIZE': 2048, 'num_warps': 8, 'num_stages': 2}` | `{'tile_size': 2048, 'occupancy': 16}` |
| bf16 | `{'batch': 1, 'M': 2048, 'K': 10240}` | `{'BLOCK_N_SIZE': 2048, 'num_warps': 8, 'num_stages': 3}` | `{'tile_size': 2048, 'occupancy': 16}` |
| fp32 | `{'batch': 1, 'M': 2048, 'K': 10240}` | `{'BLOCK_N_SIZE': 1024, 'num_warps': 8, 'num_stages': 2}` | `{'tile_size': 2048, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 20.00 us | 41.74 % | 28.97 % | 54.76 % | 25.63 % | 45.68 % | 2.22 Tbyte/s | 256 | 40 register/thread | 0 byte/block | 32 byte/block | 6 block / 14 block |
| fp16 | cutile | 35.30 us | 53.71 % | 22.78 % | 67.40 % | 45.65 % | 41.40 % | 1.74 Tbyte/s | 128 | 32 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| bf16 | triton | 20.77 us | 39.47 % | 27.93 % | 51.06 % | 24.54 % | 46.47 % | 2.14 Tbyte/s | 256 | 40 register/thread | 0 byte/block | 32 byte/block | 6 block / 14 block |
| bf16 | cutile | 37.18 us | 52.50 % | 21.60 % | 66.04 % | 43.18 % | 42.57 % | 1.66 Tbyte/s | 128 | 32 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 32.74 us | 51.14 % | 51.14 % | 56.99 % | 34.69 % | 26.52 % | 3.92 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp32 | cutile | 33.63 us | 47.79 % | 47.79 % | 51.92 % | 33.22 % | 37.41 % | 3.66 Tbyte/s | 128 | 64 register/thread | 28 byte/block | 0 byte/block | 8 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.76× faster** (20.0 µs vs 35.3 µs).
- **bf16**: Triton is **1.79× faster** (20.8 µs vs 37.2 µs).
- **fp32**: Triton is **1.03× faster** (32.7 µs vs 33.6 µs).

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
