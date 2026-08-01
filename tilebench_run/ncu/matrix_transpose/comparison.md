# NCU Comparison: matrix_transpose

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'m': 4096, 'n': 20480}` | `{'BLOCK_TILE': 64, 'num_warps': 8}` | `{'tile': 128, 'occupancy': 4}` |
| bf16 | `{'m': 4096, 'n': 20480}` | `{'BLOCK_TILE': 64, 'num_warps': 8}` | `{'tile': 128, 'occupancy': 32}` |
| fp32 | `{'m': 4096, 'n': 20480}` | `{'BLOCK_TILE': 64, 'num_warps': 8}` | `{'tile': 64, 'occupancy': 32}` |
| int8 | `{'m': 4096, 'n': 20480}` | `{'BLOCK_TILE': 128, 'num_warps': 8}` | `{'tile': 128, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 50.46 us | 72.72 % | 72.72 % | 50.13 % | 37.37 % | 24.52 % | 5.58 Tbyte/s | 256 | 26 register/thread | 0 byte/block | 8.19 Kbyte/block | 8 block / 14 block |
| fp16 | cutile | 50.14 us | 72.87 % | 72.87 % | 60.92 % | 37.55 % | 27.70 % | 5.59 Tbyte/s | 128 | 111 register/thread | 32.78 Kbyte/block | 0 byte/block | 4 block / 4 block |
| bf16 | triton | 50.14 us | 73.28 % | 73.28 % | 50.10 % | 37.66 % | 24.07 % | 5.62 Tbyte/s | 256 | 26 register/thread | 0 byte/block | 8.19 Kbyte/block | 8 block / 14 block |
| bf16 | cutile | 50.62 us | 72.12 % | 72.12 % | 61.40 % | 37.22 % | 26.26 % | 5.53 Tbyte/s | 128 | 111 register/thread | 32.78 Kbyte/block | 0 byte/block | 4 block / 4 block |
| fp32 | triton | 99.07 us | 81.11 % | 81.11 % | 48.18 % | 39.18 % | 23.85 % | 6.22 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 16.38 Kbyte/block | 8 block / 9 block |
| fp32 | cutile | 99.94 us | 80.37 % | 80.37 % | 48.09 % | 38.87 % | 23.04 % | 6.16 Tbyte/s | 128 | 55 register/thread | 16.40 Kbyte/block | 0 byte/block | 9 block / 9 block |
| int8 | triton | 28.90 us | 65.16 % | 53.65 % | 77.12 % | 31.84 % | 48.94 % | 4.11 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 16.38 Kbyte/block | 8 block / 9 block |
| int8 | cutile | 27.68 us | 64.82 % | 55.22 % | 77.31 % | 33.29 % | 56.62 % | 4.23 Tbyte/s | 128 | 64 register/thread | 16.40 Kbyte/block | 0 byte/block | 8 block / 9 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.01× faster** (50.1 µs vs 50.5 µs).
- **bf16**: Triton is **1.01× faster** (50.1 µs vs 50.6 µs).
- **fp32**: Triton is **1.01× faster** (99.1 µs vs 99.9 µs).
- **int8**: cuTile is **1.04× faster** (27.7 µs vs 28.9 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Memory is more heavily utilized than Compute
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Memory is more heavily utilized than Compute
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **int8 / cutile** — Compute and Memory are well-balanced
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
