# NCU Comparison: matmul_fp32_fp16_fp8

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'M': 4096, 'N': 4096, 'K': 20480}` | `{'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}` | `{'tm': 256, 'tn': 256, 'tk': 64, 'group_size_m': 8, 'occupancy': 4}` |
| fp16 | `{'M': 4096, 'N': 4096, 'K': 20480}` | `{'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}` | `{'tm': 256, 'tn': 256, 'tk': 64, 'group_size_m': 8, 'occupancy': 8}` |
| fp8_e4m3fn | `{'M': 4096, 'N': 4096, 'K': 20480}` | `{'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}` | `{'tm': 256, 'tn': 256, 'tk': 128, 'group_size_m': 8, 'occupancy': 8}` |
| fp8_e5m2 | `{'M': 4096, 'N': 4096, 'K': 20480}` | `{'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}` | `{'tm': 256, 'tn': 256, 'tk': 128, 'group_size_m': 8, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 6010.00 us | 92.35 % | 4.93 % | 93.62 % | 13.57 % | 11.61 % | 378.60 Gbyte/s | 256 | 170 register/thread | 0 byte/block | 196.62 Kbyte/block | 1 block / 1 block |
| fp32 | cutile | 872.03 us | 41.19 % | 17.56 % | 72.33 % | 32.38 % | 80.71 % | 1.35 Tbyte/s | 256 | 255 register/thread | 229.74 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp16 | triton | 561.76 us | 31.23 % | 13.66 % | 56.50 % | 25.81 % | 66.05 % | 1.05 Tbyte/s | 256 | 199 register/thread | 0 byte/block | 196.62 Kbyte/block | 1 block / 1 block |
| fp16 | cutile | 467.17 us | 40.93 % | 16.42 % | 72.88 % | 24.57 % | 79.92 % | 1.26 Tbyte/s | 256 | 255 register/thread | 229.60 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp8_e4m3fn | triton | 276.58 us | 30.54 % | 13.87 % | 54.87 % | 21.32 % | 64.82 % | 1.06 Tbyte/s | 256 | 192 register/thread | 0 byte/block | 196.62 Kbyte/block | 1 block / 1 block |
| fp8_e4m3fn | cutile | 223.20 us | 40.25 % | 17.21 % | 71.88 % | 26.71 % | 78.67 % | 1.32 Tbyte/s | 256 | 255 register/thread | 229.54 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp8_e5m2 | triton | 283.90 us | 30.32 % | 13.55 % | 54.90 % | 22.24 % | 64.35 % | 1.04 Tbyte/s | 256 | 192 register/thread | 0 byte/block | 196.62 Kbyte/block | 1 block / 1 block |
| fp8_e5m2 | cutile | 234.08 us | 40.05 % | 16.38 % | 71.85 % | 25.92 % | 78.26 % | 1.26 Tbyte/s | 256 | 255 register/thread | 229.54 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Key findings (auto-derived)

- **fp32**: cuTile is **6.89× faster** (872.0 µs vs 6010.0 µs).
- **fp16**: cuTile is **1.20× faster** (467.2 µs vs 561.8 µs).
- **fp8_e4m3fn**: cuTile is **1.24× faster** (223.2 µs vs 276.6 µs).
- **fp8_e5m2**: cuTile is **1.21× faster** (234.1 µs vs 283.9 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — Compute is more heavily utilized than Memory
- **fp16 / triton** — Compute is more heavily utilized than Memory
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
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
