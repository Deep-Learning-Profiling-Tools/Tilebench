# NCU Comparison: matmul_fp32_fp16_fp8

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'M': 4096, 'N': 4096, 'K': 20480}` | `{'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}` | `{'tm': 256, 'tn': 256, 'tk': 64, 'group_size_m': 8, 'occupancy': 8}` |
| fp16 | `{'M': 4096, 'N': 4096, 'K': 20480}` | `{'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}` | `{'tm': 256, 'tn': 256, 'tk': 64, 'group_size_m': 8, 'occupancy': 4}` |
| fp8_e4m3fn | `{'M': 4096, 'N': 4096, 'K': 20480}` | `{'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}` | `{'tm': 256, 'tn': 256, 'tk': 128, 'group_size_m': 8, 'occupancy': 8}` |
| fp8_e5m2 | `{'M': 4096, 'N': 4096, 'K': 20480}` | `{'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}` | `{'tm': 256, 'tn': 256, 'tk': 128, 'group_size_m': 8, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 6130.00 us | 92.00 % | 4.89 % | 93.39 % | 14.53 % | 12.29 % | 375.48 Gbyte/s | 256 | 123 register/thread | 0 byte/block | 98.32 Kbyte/block | 2 block / 2 block |
| fp32 | cutile | 981.25 us | 87.84 % | 30.21 % | 89.44 % | 64.29 % | 91.97 % | 2.32 Tbyte/s | 256 | 255 register/thread | 229.60 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp16 | triton | 671.78 us | 58.10 % | 17.81 % | 64.26 % | 48.81 % | 65.51 % | 1.37 Tbyte/s | 256 | 94 register/thread | 0 byte/block | 98.32 Kbyte/block | 2 block / 2 block |
| fp16 | cutile | 516.10 us | 85.57 % | 28.68 % | 88.43 % | 60.17 % | 89.67 % | 2.20 Tbyte/s | 256 | 255 register/thread | 229.57 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp8_e4m3fn | triton | 575.52 us | 21.80 % | 10.14 % | 25.91 % | 17.94 % | 35.63 % | 778.15 Gbyte/s | 256 | 142 register/thread | 0 byte/block | 73.74 Kbyte/block | 1 block / 3 block |
| fp8_e4m3fn | cutile | 230.78 us | 77.96 % | 25.55 % | 94.36 % | 38.53 % | 77.33 % | 1.96 Tbyte/s | 256 | 255 register/thread | 229.60 Kbyte/block | 0 byte/block | 1 block / 1 block |
| fp8_e5m2 | triton | 580.10 us | 22.10 % | 10.02 % | 25.82 % | 17.41 % | 36.12 % | 768.58 Gbyte/s | 256 | 142 register/thread | 0 byte/block | 73.74 Kbyte/block | 1 block / 3 block |
| fp8_e5m2 | cutile | 245.92 us | 78.03 % | 23.81 % | 94.41 % | 36.00 % | 77.32 % | 1.83 Tbyte/s | 256 | 255 register/thread | 229.60 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Key findings (auto-derived)

- **fp32**: cuTile is **6.25× faster** (981.2 µs vs 6130.0 µs).
- **fp16**: cuTile is **1.30× faster** (516.1 µs vs 671.8 µs).
- **fp8_e4m3fn**: cuTile is **2.49× faster** (230.8 µs vs 575.5 µs).
- **fp8_e5m2**: cuTile is **2.36× faster** (245.9 µs vs 580.1 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp16 / triton** — Compute and Memory are well-balanced
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **fp8_e4m3fn / cutile** — Compute and Memory are well-balanced
- **fp8_e4m3fn / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp8_e5m2 / cutile** — Compute and Memory are well-balanced
- **fp8_e5m2 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

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
