# NCU Comparison: swiglu

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'M': 4096, 'N': 20480}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4}` | `{'tile': 4096, 'occupancy': 8}` |
| bf16 | `{'M': 4096, 'N': 20480}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4}` | `{'tile': 4096, 'occupancy': 8}` |
| fp32 | `{'M': 4096, 'N': 20480}` | `{'BLOCK_SIZE': 1024, 'num_warps': 8}` | `{'tile': 4096, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 69.06 us | 87.48 % | 87.48 % | 35.66 % | 43.63 % | 63.14 % | 6.71 Tbyte/s | 128 | 31 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 99.52 us | 62.27 % | 62.27 % | 24.71 % | 30.42 % | 65.74 % | 4.77 Tbyte/s | 128 | 56 register/thread | 0 byte/block | 0 byte/block | 9 block / 32 block |
| bf16 | triton | 72.51 us | 85.06 % | 85.06 % | 34.26 % | 41.76 % | 62.76 % | 6.52 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | cutile | 100.38 us | 61.74 % | 61.74 % | 24.41 % | 30.17 % | 67.27 % | 4.73 Tbyte/s | 128 | 56 register/thread | 0 byte/block | 0 byte/block | 9 block / 32 block |
| fp32 | triton | 136.90 us | 91.65 % | 91.65 % | 34.77 % | 44.38 % | 33.88 % | 7.03 Tbyte/s | 256 | 22 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 147.07 us | 86.23 % | 86.23 % | 32.61 % | 41.52 % | 46.49 % | 6.61 Tbyte/s | 128 | 39 register/thread | 0 byte/block | 0 byte/block | 12 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.44× faster** (69.1 µs vs 99.5 µs).
- **bf16**: Triton is **1.38× faster** (72.5 µs vs 100.4 µs).
- **fp32**: Triton is **1.07× faster** (136.9 µs vs 147.1 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Compute and Memory are well-balanced
- **bf16 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **fp16 / cutile** — Compute and Memory are well-balanced
- **fp16 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
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
