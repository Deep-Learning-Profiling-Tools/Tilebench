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
| fp16 | triton | 24.54 us | 46.75 % | 46.75 % | 42.44 % | 22.69 % | 37.56 % | 3.58 Tbyte/s | 256 | 40 register/thread | 0 byte/block | 32 byte/block | 6 block / 14 block |
| fp16 | cutile | 37.95 us | 49.93 % | 40.37 % | 61.26 % | 42.30 % | 38.50 % | 3.09 Tbyte/s | 128 | 32 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| bf16 | triton | 24.64 us | 46.48 % | 46.48 % | 39.30 % | 22.49 % | 38.70 % | 3.56 Tbyte/s | 256 | 40 register/thread | 0 byte/block | 32 byte/block | 6 block / 14 block |
| bf16 | cutile | 39.10 us | 49.20 % | 39.31 % | 60.30 % | 40.99 % | 39.92 % | 3.01 Tbyte/s | 128 | 32 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 41.31 us | 59.37 % | 59.37 % | 45.71 % | 30.74 % | 22.35 % | 4.55 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp32 | cutile | 38.94 us | 61.14 % | 61.14 % | 42.78 % | 32.06 % | 33.54 % | 4.68 Tbyte/s | 128 | 64 register/thread | 28 byte/block | 0 byte/block | 8 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.55× faster** (24.5 µs vs 38.0 µs).
- **bf16**: Triton is **1.59× faster** (24.6 µs vs 39.1 µs).
- **fp32**: cuTile is **1.06× faster** (38.9 µs vs 41.3 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **bf16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — Memory is more heavily utilized than Compute
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
