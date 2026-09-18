# NCU Comparison: weight_dequant

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'TILE_SIZE': 128, 'M': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4}` | `{'tile': 4096, 'occupancy': 4}` |
| bf16 | `{'TILE_SIZE': 128, 'M': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4}` | `{'tile': 4096, 'occupancy': 4}` |
| fp32 | `{'TILE_SIZE': 128, 'M': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'tile': 2048, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 69.70 us | 80.16 % | 80.16 % | 40.22 % | 36.19 % | 28.70 % | 6.15 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 188.61 us | 29.95 % | 29.95 % | 26.24 % | 13.41 % | 87.52 % | 2.30 Tbyte/s | 128 | 124 register/thread | 16.40 Kbyte/block | 0 byte/block | 4 block / 7 block |
| bf16 | triton | 70.21 us | 79.51 % | 79.51 % | 40.16 % | 35.92 % | 28.15 % | 6.10 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | cutile | 188.58 us | 29.90 % | 29.90 % | 26.26 % | 13.40 % | 87.65 % | 2.29 Tbyte/s | 128 | 124 register/thread | 16.40 Kbyte/block | 0 byte/block | 4 block / 7 block |
| fp32 | triton | 126.91 us | 86.70 % | 86.70 % | 43.07 % | 39.52 % | 21.39 % | 6.65 Tbyte/s | 256 | 28 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp32 | cutile | 196.19 us | 56.56 % | 56.56 % | 46.46 % | 25.64 % | 85.40 % | 4.34 Tbyte/s | 128 | 70 register/thread | 16.40 Kbyte/block | 0 byte/block | 7 block / 7 block |

## Key findings (auto-derived)

- **fp16**: Triton is **2.71× faster** (69.7 µs vs 188.6 µs).
- **bf16**: Triton is **2.69× faster** (70.2 µs vs 188.6 µs).
- **fp32**: Triton is **1.55× faster** (126.9 µs vs 196.2 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp16 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
