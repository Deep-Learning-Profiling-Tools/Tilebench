# NCU Comparison: l2_norm

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'M': 2048, 'eps': 1e-06, 'K': 10240}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 2}` | `{'tile_size': 1024, 'occupancy': 16}` |
| bf16 | `{'batch': 1, 'M': 2048, 'eps': 1e-06, 'K': 10240}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 4}` | `{'tile_size': 1024, 'occupancy': 16}` |
| fp32 | `{'batch': 1, 'M': 2048, 'eps': 1e-06, 'K': 10240}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 2}` | `{'tile_size': 512, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 18.02 us | 59.20 % | 59.20 % | 40.97 % | 31.06 % | 20.13 % | 4.53 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp16 | cutile | 22.98 us | 51.38 % | 51.38 % | 31.72 % | 26.68 % | 30.71 % | 3.94 Tbyte/s | 128 | 29 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| bf16 | triton | 17.82 us | 61.15 % | 61.15 % | 41.39 % | 31.85 % | 21.18 % | 4.68 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| bf16 | cutile | 23.23 us | 50.71 % | 50.71 % | 30.77 % | 26.10 % | 30.68 % | 3.88 Tbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 32.48 us | 71.81 % | 71.81 % | 40.56 % | 37.05 % | 8.59 % | 5.50 Tbyte/s | 256 | 36 register/thread | 0 byte/block | 32 byte/block | 6 block / 14 block |
| fp32 | cutile | 33.66 us | 67.21 % | 67.21 % | 38.35 % | 33.58 % | 22.85 % | 5.15 Tbyte/s | 128 | 90 register/thread | 28 byte/block | 0 byte/block | 5 block / 14 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.28× faster** (18.0 µs vs 23.0 µs).
- **bf16**: Triton is **1.30× faster** (17.8 µs vs 23.2 µs).
- **fp32**: Triton is **1.04× faster** (32.5 µs vs 33.7 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — Memory is more heavily utilized than Compute
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
