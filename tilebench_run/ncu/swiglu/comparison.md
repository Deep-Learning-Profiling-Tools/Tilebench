# NCU Comparison: swiglu

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'M': 4096, 'N': 20480}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4}` | `{'tile': 8192, 'occupancy': 8}` |
| bf16 | `{'M': 4096, 'N': 20480}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4}` | `{'tile': 8192, 'occupancy': 8}` |
| fp32 | `{'M': 4096, 'N': 20480}` | `{'BLOCK_SIZE': 1024, 'num_warps': 8}` | `{'tile': 1024, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 70.02 us | 86.87 % | 86.87 % | 35.66 % | 43.05 % | 62.32 % | 6.66 Tbyte/s | 128 | 31 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 98.56 us | 62.86 % | 62.86 % | 24.94 % | 30.75 % | 63.97 % | 4.82 Tbyte/s | 128 | 56 register/thread | 0 byte/block | 0 byte/block | 9 block / 32 block |
| bf16 | triton | 72.45 us | 85.18 % | 85.18 % | 34.04 % | 41.77 % | 63.23 % | 6.53 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | cutile | 99.17 us | 62.50 % | 62.50 % | 24.83 % | 30.56 % | 65.87 % | 4.79 Tbyte/s | 128 | 56 register/thread | 0 byte/block | 0 byte/block | 9 block / 32 block |
| fp32 | triton | 138.34 us | 90.90 % | 90.90 % | 34.36 % | 44.08 % | 33.79 % | 6.97 Tbyte/s | 256 | 22 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 143.04 us | 89.06 % | 89.06 % | 33.51 % | 42.81 % | 58.64 % | 6.83 Tbyte/s | 128 | 44 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.41× faster** (70.0 µs vs 98.6 µs).
- **bf16**: Triton is **1.37× faster** (72.5 µs vs 99.2 µs).
- **fp32**: Triton is **1.03× faster** (138.3 µs vs 143.0 µs).

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
