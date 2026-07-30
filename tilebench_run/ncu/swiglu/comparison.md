# NCU Comparison: swiglu

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'M': 4096, 'N': 20480}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'tile': 16384, 'occupancy': 4}` |
| bf16 | `{'M': 4096, 'N': 20480}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4}` | `{'tile': 16384, 'occupancy': 4}` |
| fp32 | `{'M': 4096, 'N': 20480}` | `{'BLOCK_SIZE': 1024, 'num_warps': 8}` | `{'tile': 16384, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 71.17 us | 86.58 % | 86.58 % | 35.11 % | 42.62 % | 61.66 % | 6.64 Tbyte/s | 256 | 31 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp16 | cutile | 99.71 us | 62.05 % | 62.05 % | 25.08 % | 30.39 % | 62.30 % | 4.76 Tbyte/s | 128 | 56 register/thread | 0 byte/block | 0 byte/block | 9 block / 32 block |
| bf16 | triton | 72.70 us | 84.78 % | 84.78 % | 34.19 % | 41.57 % | 63.49 % | 6.50 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | cutile | 100.38 us | 61.66 % | 61.66 % | 24.76 % | 30.20 % | 64.59 % | 4.73 Tbyte/s | 128 | 56 register/thread | 0 byte/block | 0 byte/block | 9 block / 32 block |
| fp32 | triton | 136.99 us | 91.37 % | 91.37 % | 34.79 % | 44.40 % | 34.14 % | 7.01 Tbyte/s | 256 | 22 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 154.88 us | 81.84 % | 81.84 % | 31.44 % | 39.48 % | 42.38 % | 6.28 Tbyte/s | 128 | 39 register/thread | 0 byte/block | 0 byte/block | 12 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.40× faster** (71.2 µs vs 99.7 µs).
- **bf16**: Triton is **1.38× faster** (72.7 µs vs 100.4 µs).
- **fp32**: Triton is **1.13× faster** (137.0 µs vs 154.9 µs).

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
