# NCU Comparison: layernorm

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'M': 2048, 'K': 10240}` | `{'BLOCK_N_SIZE': 2048, 'num_warps': 8, 'num_stages': 2}` | `{'tile_size': 2048, 'occupancy': 8}` |
| bf16 | `{'batch': 1, 'M': 2048, 'K': 10240}` | `{'BLOCK_N_SIZE': 2048, 'num_warps': 8, 'num_stages': 4}` | `{'tile_size': 2048, 'occupancy': 8}` |
| fp32 | `{'batch': 1, 'M': 2048, 'K': 10240}` | `{'BLOCK_N_SIZE': 1024, 'num_warps': 8, 'num_stages': 4}` | `{'tile_size': 2048, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 19.68 us | 41.89 % | 29.36 % | 54.87 % | 26.17 % | 45.97 % | 2.25 Tbyte/s | 256 | 40 register/thread | 0 byte/block | 32 byte/block | 6 block / 14 block |
| fp16 | cutile | 33.22 us | 20.84 % | 17.52 % | 25.87 % | 15.65 % | 37.35 % | 1.34 Tbyte/s | 128 | 62 register/thread | 28 byte/block | 0 byte/block | 8 block / 28 block |
| bf16 | triton | 21.02 us | 39.70 % | 27.52 % | 50.33 % | 24.05 % | 46.78 % | 2.11 Tbyte/s | 256 | 40 register/thread | 0 byte/block | 32 byte/block | 6 block / 14 block |
| bf16 | cutile | 33.22 us | 20.86 % | 17.40 % | 25.06 % | 15.63 % | 37.87 % | 1.33 Tbyte/s | 128 | 62 register/thread | 28 byte/block | 0 byte/block | 8 block / 28 block |
| fp32 | triton | 33.15 us | 50.36 % | 50.36 % | 56.68 % | 33.99 % | 26.55 % | 3.86 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp32 | cutile | 33.60 us | 47.84 % | 47.84 % | 52.66 % | 33.27 % | 38.76 % | 3.67 Tbyte/s | 128 | 64 register/thread | 28 byte/block | 0 byte/block | 8 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.69× faster** (19.7 µs vs 33.2 µs).
- **bf16**: Triton is **1.58× faster** (21.0 µs vs 33.2 µs).
- **fp32**: Triton is **1.01× faster** (33.1 µs vs 33.6 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
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
