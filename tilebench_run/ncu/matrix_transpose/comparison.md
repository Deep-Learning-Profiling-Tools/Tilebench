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
| fp16 | triton | 50.18 us | 73.16 % | 73.16 % | 49.93 % | 37.66 % | 24.36 % | 5.61 Tbyte/s | 256 | 26 register/thread | 0 byte/block | 8.19 Kbyte/block | 8 block / 14 block |
| fp16 | cutile | 50.02 us | 73.03 % | 73.03 % | 61.79 % | 37.65 % | 27.37 % | 5.60 Tbyte/s | 128 | 111 register/thread | 32.78 Kbyte/block | 0 byte/block | 4 block / 4 block |
| bf16 | triton | 50.91 us | 72.11 % | 72.11 % | 49.62 % | 37.02 % | 24.31 % | 5.53 Tbyte/s | 256 | 26 register/thread | 0 byte/block | 8.19 Kbyte/block | 8 block / 14 block |
| bf16 | cutile | 50.56 us | 72.25 % | 72.25 % | 60.83 % | 37.28 % | 27.45 % | 5.54 Tbyte/s | 128 | 111 register/thread | 32.78 Kbyte/block | 0 byte/block | 4 block / 4 block |
| fp32 | triton | 98.66 us | 81.44 % | 81.44 % | 48.30 % | 39.37 % | 23.64 % | 6.25 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 16.38 Kbyte/block | 8 block / 9 block |
| fp32 | cutile | 102.34 us | 78.46 % | 78.46 % | 44.53 % | 37.95 % | 17.98 % | 6.02 Tbyte/s | 128 | 160 register/thread | 65.55 Kbyte/block | 0 byte/block | 3 block / 3 block |
| int8 | triton | 29.06 us | 65.66 % | 53.46 % | 77.82 % | 31.67 % | 49.28 % | 4.09 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 16.38 Kbyte/block | 8 block / 9 block |
| int8 | cutile | 34.27 us | 52.09 % | 46.39 % | 60.62 % | 26.79 % | 46.57 % | 3.55 Tbyte/s | 128 | 113 register/thread | 16.40 Kbyte/block | 0 byte/block | 4 block / 7 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.00× faster** (50.0 µs vs 50.2 µs).
- **bf16**: cuTile is **1.01× faster** (50.6 µs vs 50.9 µs).
- **fp32**: Triton is **1.04× faster** (98.7 µs vs 102.3 µs).
- **int8**: Triton is **1.18× faster** (29.1 µs vs 34.3 µs).

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
