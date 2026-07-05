# NCU Comparison: matrix_transpose

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'m': 4096, 'n': 20480}` | `{'BLOCK_TILE': 64, 'num_warps': 8}` | `{'tile': 128, 'occupancy': 4}` |
| bf16 | `{'m': 4096, 'n': 20480}` | `{'BLOCK_TILE': 64, 'num_warps': 8}` | `{'tile': 128, 'occupancy': 4}` |
| fp32 | `{'m': 4096, 'n': 20480}` | `{'BLOCK_TILE': 64, 'num_warps': 8}` | `{'tile': 128, 'occupancy': 4}` |
| int8 | `{'m': 4096, 'n': 20480}` | `{'BLOCK_TILE': 128, 'num_warps': 8}` | `{'tile': 128, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 50.50 us | 72.85 % | 72.85 % | 49.84 % | 37.39 % | 24.54 % | 5.58 Tbyte/s | 256 | 26 register/thread | 0 byte/block | 8.19 Kbyte/block | 8 block / 14 block |
| fp16 | cutile | 50.82 us | 71.91 % | 71.91 % | 61.19 % | 37.16 % | 27.42 % | 5.51 Tbyte/s | 128 | 111 register/thread | 32.78 Kbyte/block | 0 byte/block | 4 block / 4 block |
| bf16 | triton | 50.14 us | 73.25 % | 73.25 % | 50.19 % | 37.67 % | 24.04 % | 5.61 Tbyte/s | 256 | 26 register/thread | 0 byte/block | 8.19 Kbyte/block | 8 block / 14 block |
| bf16 | cutile | 50.46 us | 72.42 % | 72.42 % | 61.94 % | 37.39 % | 27.69 % | 5.55 Tbyte/s | 128 | 111 register/thread | 32.78 Kbyte/block | 0 byte/block | 4 block / 4 block |
| fp32 | triton | 99.46 us | 80.82 % | 80.82 % | 48.21 % | 39.09 % | 23.75 % | 6.20 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 16.38 Kbyte/block | 8 block / 9 block |
| fp32 | cutile | 101.70 us | 78.95 % | 78.95 % | 44.16 % | 38.22 % | 17.89 % | 6.05 Tbyte/s | 128 | 160 register/thread | 65.55 Kbyte/block | 0 byte/block | 3 block / 3 block |
| int8 | triton | 28.74 us | 63.69 % | 53.98 % | 77.31 % | 31.95 % | 47.82 % | 4.14 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 16.38 Kbyte/block | 8 block / 9 block |
| int8 | cutile | 34.02 us | 52.96 % | 46.73 % | 60.92 % | 27.01 % | 47.14 % | 3.58 Tbyte/s | 128 | 113 register/thread | 16.40 Kbyte/block | 0 byte/block | 4 block / 7 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.01× faster** (50.5 µs vs 50.8 µs).
- **bf16**: Triton is **1.01× faster** (50.1 µs vs 50.5 µs).
- **fp32**: Triton is **1.02× faster** (99.5 µs vs 101.7 µs).
- **int8**: Triton is **1.18× faster** (28.7 µs vs 34.0 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Memory is more heavily utilized than Compute
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Memory is more heavily utilized than Compute
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — Memory is more heavily utilized than Compute
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **int8 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **int8 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `cutile_int8.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`
- `triton_int8.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
