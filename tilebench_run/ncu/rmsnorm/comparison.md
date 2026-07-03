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
| fp16 | triton | 17.89 us | 33.45 % | 32.39 % | 45.27 % | 30.13 % | 34.26 % | 2.48 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp16 | cutile | 20.61 us | 28.17 % | 28.17 % | 35.99 % | 26.51 % | 47.97 % | 2.16 Tbyte/s | 128 | 31 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| bf16 | triton | 17.86 us | 34.11 % | 32.40 % | 45.38 % | 30.25 % | 37.80 % | 2.48 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| bf16 | cutile | 21.34 us | 27.19 % | 27.19 % | 34.80 % | 25.66 % | 50.15 % | 2.08 Tbyte/s | 128 | 31 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 30.78 us | 53.84 % | 53.84 % | 48.57 % | 36.12 % | 22.73 % | 4.13 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp32 | cutile | 36.93 us | 64.54 % | 64.54 % | 36.49 % | 37.19 % | 23.07 % | 4.95 Tbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.15× faster** (17.9 µs vs 20.6 µs).
- **bf16**: Triton is **1.19× faster** (17.9 µs vs 21.3 µs).
- **fp32**: Triton is **1.20× faster** (30.8 µs vs 36.9 µs).

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
