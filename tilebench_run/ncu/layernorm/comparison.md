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
| fp16 | triton | 19.81 us | 41.36 % | 29.30 % | 55.34 % | 25.91 % | 45.31 % | 2.24 Tbyte/s | 256 | 40 register/thread | 0 byte/block | 32 byte/block | 6 block / 14 block |
| fp16 | cutile | 32.93 us | 21.31 % | 17.57 % | 25.48 % | 15.75 % | 38.14 % | 1.35 Tbyte/s | 128 | 62 register/thread | 28 byte/block | 0 byte/block | 8 block / 28 block |
| bf16 | triton | 20.86 us | 38.60 % | 27.75 % | 51.50 % | 24.34 % | 45.39 % | 2.12 Tbyte/s | 256 | 40 register/thread | 0 byte/block | 32 byte/block | 6 block / 14 block |
| bf16 | cutile | 33.38 us | 21.14 % | 17.34 % | 25.98 % | 15.61 % | 38.10 % | 1.33 Tbyte/s | 128 | 62 register/thread | 28 byte/block | 0 byte/block | 8 block / 28 block |
| fp32 | triton | 32.96 us | 50.94 % | 50.94 % | 56.67 % | 34.52 % | 26.37 % | 3.90 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp32 | cutile | 33.34 us | 48.52 % | 48.52 % | 52.96 % | 33.62 % | 38.67 % | 3.72 Tbyte/s | 128 | 64 register/thread | 28 byte/block | 0 byte/block | 8 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.66× faster** (19.8 µs vs 32.9 µs).
- **bf16**: Triton is **1.60× faster** (20.9 µs vs 33.4 µs).
- **fp32**: Triton is **1.01× faster** (33.0 µs vs 33.3 µs).

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
