# NCU Comparison: mean_reduction

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_M': 1, 'BLOCK_N': 1024, 'num_warps': 8}` | `{'tile_size': 512, 'occupancy': 4}` |
| bf16 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_M': 1, 'BLOCK_N': 2048, 'num_warps': 8}` | `{'tile_size': 1024, 'occupancy': 16}` |
| fp32 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_M': 1, 'BLOCK_N': 512, 'num_warps': 8}` | `{'tile_size': 1024, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 16.70 us | 67.52 % | 67.52 % | 24.72 % | 43.80 % | 21.30 % | 5.16 Tbyte/s | 256 | 31 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp16 | cutile | 19.74 us | 57.02 % | 57.02 % | 24.67 % | 37.38 % | 39.65 % | 4.36 Tbyte/s | 128 | 74 register/thread | 28 byte/block | 0 byte/block | 6 block / 14 block |
| bf16 | triton | 16.74 us | 67.21 % | 67.21 % | 24.97 % | 43.71 % | 21.28 % | 5.14 Tbyte/s | 256 | 30 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| bf16 | cutile | 19.55 us | 57.53 % | 57.53 % | 18.63 % | 37.43 % | 31.23 % | 4.40 Tbyte/s | 128 | 28 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 29.89 us | 74.87 % | 74.87 % | 22.34 % | 48.98 % | 10.46 % | 5.73 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp32 | cutile | 30.24 us | 74.00 % | 74.00 % | 20.16 % | 48.47 % | 21.47 % | 5.67 Tbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.18× faster** (16.7 µs vs 19.7 µs).
- **bf16**: Triton is **1.17× faster** (16.7 µs vs 19.6 µs).
- **fp32**: Triton is **1.01× faster** (29.9 µs vs 30.2 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — Memory is more heavily utilized than Compute
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
