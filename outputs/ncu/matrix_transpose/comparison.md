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
| fp16 | triton | 56.38 us | 78.06 % | 78.06 % | 43.53 % | 35.63 % | 21.70 % | 5.98 Tbyte/s | 256 | 26 register/thread | 0 byte/block | 8.19 Kbyte/block | 8 block / 14 block |
| fp16 | cutile | 56.77 us | 77.15 % | 77.15 % | 52.25 % | 35.29 % | 24.17 % | 5.91 Tbyte/s | 128 | 111 register/thread | 32.78 Kbyte/block | 0 byte/block | 4 block / 4 block |
| bf16 | triton | 57.09 us | 77.05 % | 77.05 % | 42.87 % | 35.12 % | 21.56 % | 5.91 Tbyte/s | 256 | 26 register/thread | 0 byte/block | 8.19 Kbyte/block | 8 block / 14 block |
| bf16 | cutile | 55.94 us | 78.24 % | 78.24 % | 52.08 % | 35.79 % | 24.17 % | 6.00 Tbyte/s | 128 | 111 register/thread | 32.78 Kbyte/block | 0 byte/block | 4 block / 4 block |
| fp32 | triton | 106.56 us | 82.22 % | 82.22 % | 43.97 % | 37.59 % | 22.01 % | 6.31 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 16.38 Kbyte/block | 8 block / 9 block |
| fp32 | cutile | 107.10 us | 81.76 % | 81.76 % | 43.51 % | 37.35 % | 21.29 % | 6.27 Tbyte/s | 128 | 55 register/thread | 16.40 Kbyte/block | 0 byte/block | 9 block / 9 block |
| int8 | triton | 32.74 us | 67.02 % | 67.02 % | 65.54 % | 30.71 % | 42.29 % | 5.14 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 16.38 Kbyte/block | 8 block / 9 block |
| int8 | cutile | 32.54 us | 67.21 % | 67.21 % | 64.33 % | 30.91 % | 48.63 % | 5.15 Tbyte/s | 128 | 64 register/thread | 16.40 Kbyte/block | 0 byte/block | 8 block / 9 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.01× faster** (56.4 µs vs 56.8 µs).
- **bf16**: cuTile is **1.02× faster** (55.9 µs vs 57.1 µs).
- **fp32**: Triton is **1.01× faster** (106.6 µs vs 107.1 µs).
- **int8**: cuTile is **1.01× faster** (32.5 µs vs 32.7 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Memory is more heavily utilized than Compute
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Memory is more heavily utilized than Compute
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **int8 / cutile** — Memory is more heavily utilized than Compute
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
