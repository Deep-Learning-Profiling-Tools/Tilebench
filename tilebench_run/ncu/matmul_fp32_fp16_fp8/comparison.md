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
| fp32 | triton | 6000.00 us | 92.37 % | 4.94 % | 93.66 % | 13.45 % | 11.61 % | 379.30 Gbyte/s | 256 | 170 register/thread | 0 byte/block | 196.62 Kbyte/block | 1 block / 1 block |
| fp32 | cutile | 869.76 us | 41.28 % | 17.61 % | 72.31 % | 33.35 % | 80.94 % | 1.35 Tbyte/s | 256 | 255 register/thread | 229.74 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp16 | triton | 554.62 us | 31.63 % | 13.98 % | 57.01 % | 22.42 % | 66.82 % | 1.07 Tbyte/s | 128 | 255 register/thread | 0 byte/block | 196.62 Kbyte/block | 2 block / 1 block |
| fp16 | cutile | 462.88 us | 41.07 % | 16.58 % | 72.84 % | 26.72 % | 80.20 % | 1.27 Tbyte/s | 256 | 255 register/thread | 229.60 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp8_e4m3fn | triton | 70.40 us | 74.74 % | 74.74 % | 20.84 % | 41.37 % | 65.50 % | 5.73 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp8_e4m3fn | cutile | 70.85 us | 74.22 % | 74.22 % | 20.86 % | 41.12 % | 66.45 % | 5.69 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp8_e5m2 | triton | 71.10 us | 74.06 % | 74.06 % | 20.67 % | 40.98 % | 67.75 % | 5.68 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp8_e5m2 | cutile | 71.30 us | 73.85 % | 73.85 % | 20.75 % | 40.86 % | 67.51 % | 5.66 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |

## Key findings (auto-derived)

- **fp32**: cuTile is **6.90× faster** (869.8 µs vs 6000.0 µs).
- **fp16**: cuTile is **1.20× faster** (462.9 µs vs 554.6 µs).
- **fp8_e4m3fn**: Triton is **1.01× faster** (70.4 µs vs 70.8 µs).
- **fp8_e5m2**: Triton is **1.00× faster** (71.1 µs vs 71.3 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp16 / triton** — Compute is more heavily utilized than Memory
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
