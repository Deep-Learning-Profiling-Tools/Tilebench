# NCU Comparison: l2_norm

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'M': 2048, 'eps': 1e-06, 'K': 10240}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 3}` | `{'tile_size': 1024, 'occupancy': 32}` |
| bf16 | `{'batch': 1, 'M': 2048, 'eps': 1e-06, 'K': 10240}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 4}` | `{'tile_size': 1024, 'occupancy': 32}` |
| fp32 | `{'batch': 1, 'M': 2048, 'eps': 1e-06, 'K': 10240}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 4}` | `{'tile_size': 1024, 'occupancy': 32}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 16.64 us | 34.96 % | 34.96 % | 46.92 % | 30.26 % | 21.71 % | 2.68 Tbyte/s | 128 | 48 register/thread | 0 byte/block | 16 byte/block | 10 block / 28 block |
| fp16 | cutile | 19.84 us | 29.65 % | 29.65 % | 41.29 % | 27.21 % | 36.78 % | 2.27 Tbyte/s | 128 | 24 register/thread | 28 byte/block | 0 byte/block | 21 block / 28 block |
| bf16 | triton | 16.74 us | 34.75 % | 34.75 % | 46.73 % | 30.08 % | 19.49 % | 2.66 Tbyte/s | 128 | 48 register/thread | 0 byte/block | 16 byte/block | 10 block / 28 block |
| bf16 | cutile | 19.04 us | 30.64 % | 30.64 % | 40.29 % | 28.05 % | 37.53 % | 2.34 Tbyte/s | 128 | 24 register/thread | 28 byte/block | 0 byte/block | 21 block / 28 block |
| fp32 | triton | 34.56 us | 63.33 % | 63.33 % | 38.91 % | 37.59 % | 7.29 % | 4.86 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp32 | cutile | 42.91 us | 59.82 % | 59.82 % | 50.68 % | 39.47 % | 15.16 % | 4.58 Tbyte/s | 128 | 24 register/thread | 28 byte/block | 0 byte/block | 21 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.19× faster** (16.6 µs vs 19.8 µs).
- **bf16**: Triton is **1.14× faster** (16.7 µs vs 19.0 µs).
- **fp32**: Triton is **1.24× faster** (34.6 µs vs 42.9 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **bf16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **fp32 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
