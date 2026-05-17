# NCU Comparison: weight_dequant

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'TILE_SIZE': 128, 'M': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4}` | `{'tile': 2048, 'occupancy': 16}` |
| bf16 | `{'TILE_SIZE': 128, 'M': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4}` | `{'tile': 2048, 'occupancy': 16}` |
| fp32 | `{'TILE_SIZE': 128, 'M': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'tile': 2048, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 62.02 us | 78.09 % | 78.09 % | 46.79 % | 38.73 % | 32.03 % | 5.99 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 326.82 us | 18.81 % | 14.93 % | 19.06 % | 7.35 % | 85.19 % | 1.15 Tbyte/s | 128 | 32 register/thread | 12.30 Kbyte/block | 0 byte/block | 16 block / 17 block |
| bf16 | triton | 62.30 us | 77.70 % | 77.70 % | 46.53 % | 38.54 % | 31.83 % | 5.96 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | cutile | 328.67 us | 18.66 % | 14.84 % | 18.91 % | 7.30 % | 85.00 % | 1.14 Tbyte/s | 128 | 32 register/thread | 12.30 Kbyte/block | 0 byte/block | 16 block / 17 block |
| fp32 | triton | 117.06 us | 87.46 % | 87.46 % | 46.00 % | 41.77 % | 22.74 % | 6.71 Tbyte/s | 256 | 28 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp32 | cutile | 326.02 us | 32.26 % | 31.97 % | 32.72 % | 19.03 % | 86.96 % | 2.45 Tbyte/s | 128 | 32 register/thread | 8.20 Kbyte/block | 0 byte/block | 16 block / 17 block |

## Key findings (auto-derived)

- **fp16**: Triton is **5.27× faster** (62.0 µs vs 326.8 µs).
- **bf16**: Triton is **5.28× faster** (62.3 µs vs 328.7 µs).
- **fp32**: Triton is **2.79× faster** (117.1 µs vs 326.0 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp16 / triton** — Memory is more heavily utilized than Compute
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
