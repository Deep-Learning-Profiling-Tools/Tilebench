# NCU Comparison: matmul_fp32_fp16_fp8

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'M': 4096, 'N': 4096, 'K': 20480}` | `{'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}` | `{'tm': 256, 'tn': 256, 'tk': 64, 'group_size_m': 8, 'occupancy': 8}` |
| fp16 | `{'M': 4096, 'N': 4096, 'K': 20480}` | `{'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}` | `{'tm': 256, 'tn': 256, 'tk': 64, 'group_size_m': 8, 'occupancy': 8}` |
| fp8_e4m3fn | `{'M': 4096, 'N': 4096, 'K': 20480}` | `{'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}` | `{'tm': 256, 'tn': 256, 'tk': 128, 'group_size_m': 8, 'occupancy': 4}` |
| fp8_e5m2 | `{'M': 4096, 'N': 4096, 'K': 20480}` | `{'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}` | `{'tm': 256, 'tn': 256, 'tk': 128, 'group_size_m': 8, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 6240.00 us | 90.15 % | 4.79 % | 91.46 % | 14.36 % | 11.91 % | 367.72 Gbyte/s | 256 | 124 register/thread | 0 byte/block | 98.32 Kbyte/block | 2 block / 2 block |
| fp32 | cutile | 965.79 us | 86.21 % | 30.69 % | 88.95 % | 62.88 % | 90.66 % | 2.35 Tbyte/s | 256 | 255 register/thread | 229.60 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp16 | triton | 671.87 us | 55.58 % | 17.82 % | 61.49 % | 49.56 % | 62.73 % | 1.37 Tbyte/s | 256 | 90 register/thread | 0 byte/block | 98.32 Kbyte/block | 2 block / 2 block |
| fp16 | cutile | 507.42 us | 84.52 % | 29.18 % | 87.38 % | 61.53 % | 88.86 % | 2.24 Tbyte/s | 256 | 255 register/thread | 229.57 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp8_e4m3fn | triton | 70.50 us | 74.65 % | 74.65 % | 20.87 % | 41.36 % | 66.53 % | 5.72 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp8_e4m3fn | cutile | 70.50 us | 74.63 % | 74.63 % | 20.84 % | 41.32 % | 66.53 % | 5.72 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp8_e5m2 | triton | 71.20 us | 73.94 % | 73.94 % | 20.65 % | 40.91 % | 67.44 % | 5.67 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp8_e5m2 | cutile | 71.42 us | 73.71 % | 73.71 % | 20.83 % | 40.80 % | 67.29 % | 5.65 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |

## Key findings (auto-derived)

- **fp32**: cuTile is **6.46× faster** (965.8 µs vs 6240.0 µs).
- **fp16**: cuTile is **1.32× faster** (507.4 µs vs 671.9 µs).
- **fp8_e4m3fn**: cuTile is **1.00× faster** (70.5 µs vs 70.5 µs).
- **fp8_e5m2**: Triton is **1.00× faster** (71.2 µs vs 71.4 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp16 / triton** — Compute and Memory are well-balanced
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **fp8_e4m3fn / cutile** — Compute and Memory are well-balanced
- **fp8_e4m3fn / triton** — Compute and Memory are well-balanced
- **fp8_e5m2 / cutile** — Compute and Memory are well-balanced
- **fp8_e5m2 / triton** — Compute and Memory are well-balanced

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
