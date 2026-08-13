# NCU Comparison: leaky_relu

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 50000000}` | `{'BLOCK_SIZE': 4096, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 4096, 'occupancy': 16}` |
| bf16 | `{'n': 50000000}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 4096, 'occupancy': 8}` |
| fp32 | `{'n': 50000000}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8, 'num_stages': 2}` | `{'tile': 4096, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 26.82 us | 70.09 % | 70.09 % | 55.54 % | 40.47 % | 27.85 % | 5.37 Tbyte/s | 256 | 24 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp16 | cutile | 27.23 us | 69.01 % | 69.01 % | 56.31 % | 39.63 % | 49.79 % | 5.28 Tbyte/s | 128 | 28 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | triton | 26.43 us | 71.12 % | 71.12 % | 56.12 % | 41.05 % | 27.63 % | 5.45 Tbyte/s | 128 | 24 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| bf16 | cutile | 26.72 us | 70.32 % | 70.32 % | 56.25 % | 40.43 % | 50.57 % | 5.39 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | triton | 55.78 us | 81.30 % | 81.30 % | 47.21 % | 40.78 % | 12.48 % | 6.23 Tbyte/s | 256 | 18 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 56.06 us | 80.38 % | 80.38 % | 47.62 % | 40.50 % | 25.91 % | 6.16 Tbyte/s | 128 | 51 register/thread | 0 byte/block | 0 byte/block | 9 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.02× faster** (26.8 µs vs 27.2 µs).
- **bf16**: Triton is **1.01× faster** (26.4 µs vs 26.7 µs).
- **fp32**: Triton is **1.01× faster** (55.8 µs vs 56.1 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Memory is more heavily utilized than Compute
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Memory is more heavily utilized than Compute
- **fp16 / triton** — Memory is more heavily utilized than Compute
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
