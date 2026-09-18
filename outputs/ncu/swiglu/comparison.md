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
| fp16 | triton | 77.38 us | 87.50 % | 87.50 % | 31.82 % | 40.41 % | 56.11 % | 6.71 Tbyte/s | 128 | 31 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 103.55 us | 66.91 % | 66.91 % | 23.59 % | 30.42 % | 60.72 % | 5.13 Tbyte/s | 128 | 56 register/thread | 0 byte/block | 0 byte/block | 9 block / 32 block |
| bf16 | triton | 78.88 us | 86.94 % | 86.94 % | 31.32 % | 39.73 % | 57.80 % | 6.67 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | cutile | 104.16 us | 66.52 % | 66.52 % | 23.52 % | 30.23 % | 62.85 % | 5.10 Tbyte/s | 128 | 56 register/thread | 0 byte/block | 0 byte/block | 9 block / 32 block |
| fp32 | triton | 145.89 us | 91.16 % | 91.16 % | 32.31 % | 42.52 % | 31.67 % | 6.99 Tbyte/s | 256 | 22 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 149.70 us | 89.85 % | 89.85 % | 32.18 % | 41.57 % | 56.18 % | 6.89 Tbyte/s | 128 | 44 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.34× faster** (77.4 µs vs 103.5 µs).
- **bf16**: Triton is **1.32× faster** (78.9 µs vs 104.2 µs).
- **fp32**: Triton is **1.03× faster** (145.9 µs vs 149.7 µs).

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
