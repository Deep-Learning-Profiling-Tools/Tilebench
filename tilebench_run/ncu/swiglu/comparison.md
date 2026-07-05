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
| fp16 | triton | 68.96 us | 87.40 % | 87.40 % | 36.05 % | 43.57 % | 63.37 % | 6.70 Tbyte/s | 128 | 31 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 99.81 us | 62.13 % | 62.13 % | 24.57 % | 30.37 % | 65.57 % | 4.76 Tbyte/s | 128 | 56 register/thread | 0 byte/block | 0 byte/block | 9 block / 32 block |
| bf16 | triton | 72.93 us | 84.62 % | 84.62 % | 33.98 % | 41.56 % | 63.62 % | 6.49 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | cutile | 100.80 us | 61.48 % | 61.48 % | 24.42 % | 30.04 % | 67.79 % | 4.72 Tbyte/s | 128 | 56 register/thread | 0 byte/block | 0 byte/block | 9 block / 32 block |
| fp32 | triton | 137.86 us | 91.29 % | 91.29 % | 34.49 % | 44.28 % | 33.91 % | 7.00 Tbyte/s | 256 | 22 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 148.26 us | 85.57 % | 85.57 % | 32.58 % | 41.25 % | 46.82 % | 6.56 Tbyte/s | 128 | 39 register/thread | 0 byte/block | 0 byte/block | 12 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.45× faster** (69.0 µs vs 99.8 µs).
- **bf16**: Triton is **1.38× faster** (72.9 µs vs 100.8 µs).
- **fp32**: Triton is **1.08× faster** (137.9 µs vs 148.3 µs).

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
