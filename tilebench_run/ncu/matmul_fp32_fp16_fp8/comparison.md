# NCU Comparison: matmul_fp32_fp16_fp8

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'M': 4096, 'N': 4096, 'K': 20480}` | `{'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}` | `{'tm': 256, 'tn': 256, 'tk': 64, 'group_size_m': 8, 'occupancy': 8}` |
| fp16 | `{'M': 4096, 'N': 4096, 'K': 20480}` | `{'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}` | `{'tm': 256, 'tn': 256, 'tk': 64, 'group_size_m': 8, 'occupancy': 8}` |
| fp8_e4m3fn | `{'M': 4096, 'N': 4096, 'K': 20480}` | `{'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}` | `{'tm': 256, 'tn': 256, 'tk': 128, 'group_size_m': 8, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 1340.00 us | 53.91 % | 18.38 % | 60.77 % | 49.44 % | 60.80 % | 1.41 Tbyte/s | 128 | 135 register/thread | 0 byte/block | 98.35 Kbyte/block | 3 block / 2 block |
| fp32 | cutile | 871.71 us | 41.66 % | 17.56 % | 72.45 % | 32.36 % | 81.44 % | 1.35 Tbyte/s | 256 | 255 register/thread | 229.74 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp16 | triton | 517.95 us | 36.08 % | 14.89 % | 64.24 % | 28.13 % | 72.94 % | 1.14 Tbyte/s | 128 | 255 register/thread | 0 byte/block | 196.66 Kbyte/block | 2 block / 1 block |
| fp16 | cutile | 467.17 us | 41.30 % | 16.43 % | 72.87 % | 25.77 % | 80.65 % | 1.26 Tbyte/s | 256 | 255 register/thread | 229.60 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp8_e4m3fn | triton | 257.06 us | 34.64 % | 14.94 % | 61.58 % | 26.71 % | 70.61 % | 1.15 Tbyte/s | 128 | 255 register/thread | 0 byte/block | 196.66 Kbyte/block | 2 block / 1 block |
| fp8_e4m3fn | cutile | 220.77 us | 40.13 % | 17.36 % | 71.79 % | 28.30 % | 78.43 % | 1.33 Tbyte/s | 256 | 255 register/thread | 229.54 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Key findings (auto-derived)

- **fp32**: cuTile is **1.54× faster** (871.7 µs vs 1340.0 µs).
- **fp16**: cuTile is **1.11× faster** (467.2 µs vs 518.0 µs).
- **fp8_e4m3fn**: cuTile is **1.16× faster** (220.8 µs vs 257.1 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp16 / triton** — Compute is more heavily utilized than Memory
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp32 / triton** — Compute and Memory are well-balanced
- **fp8_e4m3fn / cutile** — Compute is more heavily utilized than Memory
- **fp8_e4m3fn / triton** — Compute is more heavily utilized than Memory
- **fp8_e5m2 / cutile** — Compute is more heavily utilized than Memory
- **fp8_e5m2 / triton** — Compute is more heavily utilized than Memory

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `cutile_fp8_e4m3fn.ncu-rep`
- `cutile_fp8_e5m2.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`
- `triton_fp8_e4m3fn.ncu-rep`
- `triton_fp8_e5m2.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
