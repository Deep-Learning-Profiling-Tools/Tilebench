# NCU Comparison: rmsnorm

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'M': 2048, 'K': 10240}` | `{'BLOCK_N_SIZE': 2048, 'num_warps': 4, 'num_stages': 2}` | `{'tile_size': 1024, 'occupancy': 16}` |
| bf16 | `{'batch': 1, 'M': 2048, 'K': 10240}` | `{'BLOCK_N_SIZE': 2048, 'num_warps': 4, 'num_stages': 2}` | `{'tile_size': 1024, 'occupancy': 16}` |
| fp32 | `{'batch': 1, 'M': 2048, 'K': 10240}` | `{'BLOCK_N_SIZE': 2048, 'num_warps': 8, 'num_stages': 2}` | `{'tile_size': 1024, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 17.76 us | 33.58 % | 32.71 % | 45.34 % | 30.39 % | 34.39 % | 2.50 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp16 | cutile | 21.66 us | 27.37 % | 26.89 % | 35.88 % | 25.07 % | 48.44 % | 2.06 Tbyte/s | 128 | 31 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| bf16 | triton | 17.86 us | 34.06 % | 32.42 % | 44.88 % | 30.31 % | 37.70 % | 2.48 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| bf16 | cutile | 21.92 us | 26.68 % | 26.68 % | 33.67 % | 24.94 % | 49.96 % | 2.04 Tbyte/s | 128 | 31 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 30.24 us | 54.55 % | 54.55 % | 48.80 % | 36.79 % | 22.59 % | 4.18 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp32 | cutile | 37.28 us | 63.88 % | 63.88 % | 36.76 % | 36.77 % | 23.07 % | 4.89 Tbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.22× faster** (17.8 µs vs 21.7 µs).
- **bf16**: Triton is **1.23× faster** (17.9 µs vs 21.9 µs).
- **fp32**: Triton is **1.23× faster** (30.2 µs vs 37.3 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **bf16 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
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
