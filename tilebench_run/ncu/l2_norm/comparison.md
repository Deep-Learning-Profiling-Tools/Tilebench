# NCU Comparison: l2_norm

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'M': 2048, 'eps': 1e-06, 'K': 10240}` | `{'BLOCK_N': 2048, 'num_warps': 4, 'num_stages': 3}` | `{'tile_size': 1024, 'occupancy': 32}` |
| bf16 | `{'batch': 1, 'M': 2048, 'eps': 1e-06, 'K': 10240}` | `{'BLOCK_N': 1024, 'num_warps': 8, 'num_stages': 4}` | `{'tile_size': 1024, 'occupancy': 32}` |
| fp32 | `{'batch': 1, 'M': 2048, 'eps': 1e-06, 'K': 10240}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 4}` | `{'tile_size': 1024, 'occupancy': 32}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 17.25 us | 33.50 % | 33.50 % | 44.45 % | 30.46 % | 24.50 % | 2.56 Tbyte/s | 128 | 40 register/thread | 0 byte/block | 16 byte/block | 12 block / 28 block |
| fp16 | cutile | 18.46 us | 31.28 % | 31.28 % | 41.16 % | 29.10 % | 38.10 % | 2.40 Tbyte/s | 128 | 24 register/thread | 28 byte/block | 0 byte/block | 21 block / 28 block |
| bf16 | triton | 17.25 us | 33.52 % | 33.52 % | 44.40 % | 29.80 % | 39.87 % | 2.57 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| bf16 | cutile | 18.11 us | 31.99 % | 31.99 % | 41.15 % | 29.72 % | 37.65 % | 2.45 Tbyte/s | 128 | 24 register/thread | 28 byte/block | 0 byte/block | 21 block / 28 block |
| fp32 | triton | 30.30 us | 54.53 % | 54.53 % | 46.99 % | 36.19 % | 18.56 % | 4.18 Tbyte/s | 256 | 26 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp32 | cutile | 42.53 us | 60.23 % | 60.23 % | 51.04 % | 39.81 % | 14.26 % | 4.62 Tbyte/s | 128 | 24 register/thread | 28 byte/block | 0 byte/block | 21 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.07× faster** (17.2 µs vs 18.5 µs).
- **bf16**: Triton is **1.05× faster** (17.2 µs vs 18.1 µs).
- **fp32**: Triton is **1.40× faster** (30.3 µs vs 42.5 µs).

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
