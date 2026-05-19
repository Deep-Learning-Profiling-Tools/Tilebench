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
| fp32 | triton | 95.87 us | 83.96 % | 83.96 % | 45.56 % | 40.62 % | 9.69 % | 6.44 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | cutile | 94.94 us | 84.74 % | 84.74 % | 45.48 % | 41.03 % | 9.50 % | 6.50 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | triton | 51.36 us | 73.45 % | 73.45 % | 43.24 % | 37.09 % | 21.52 % | 5.63 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 51.62 us | 73.07 % | 73.07 % | 42.93 % | 36.95 % | 21.66 % | 5.60 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp8_e4m3fn | triton | 168.22 us | 22.15 % | 21.35 % | 22.99 % | 17.35 % | 72.29 % | 1.64 Tbyte/s | 256 | 56 register/thread | 0 byte/block | 0 byte/block | 4 block / 32 block |
| fp8_e4m3fn | cutile | 168.06 us | 22.18 % | 21.38 % | 22.98 % | 17.38 % | 72.42 % | 1.64 Tbyte/s | 256 | 56 register/thread | 0 byte/block | 0 byte/block | 4 block / 32 block |
| fp8_e5m2 | triton | 168.42 us | 22.19 % | 21.33 % | 22.99 % | 17.33 % | 72.43 % | 1.64 Tbyte/s | 256 | 56 register/thread | 0 byte/block | 0 byte/block | 4 block / 32 block |
| fp8_e5m2 | cutile | 168.22 us | 22.22 % | 21.37 % | 22.96 % | 17.37 % | 72.55 % | 1.64 Tbyte/s | 256 | 56 register/thread | 0 byte/block | 0 byte/block | 4 block / 32 block |

## Key findings (auto-derived)

- **fp32**: cuTile is **1.01× faster** (94.9 µs vs 95.9 µs).
- **fp16**: Triton is **1.01× faster** (51.4 µs vs 51.6 µs).
- **fp8_e4m3fn**: cuTile is **1.00× faster** (168.1 µs vs 168.2 µs).
- **fp8_e5m2**: cuTile is **1.00× faster** (168.2 µs vs 168.4 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — Memory is more heavily utilized than Compute
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
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
