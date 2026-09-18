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
| fp16 | triton | 33.50 us | 78.02 % | 78.02 % | 40.68 % | 35.80 % | 21.59 % | 5.98 Tbyte/s | 256 | 24 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp16 | cutile | 34.34 us | 76.44 % | 76.44 % | 41.24 % | 35.02 % | 38.26 % | 5.85 Tbyte/s | 128 | 28 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | triton | 34.56 us | 76.10 % | 76.10 % | 40.24 % | 34.81 % | 21.41 % | 5.83 Tbyte/s | 128 | 24 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| bf16 | cutile | 35.01 us | 74.85 % | 74.85 % | 41.21 % | 34.34 % | 39.06 % | 5.73 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | triton | 62.24 us | 84.09 % | 84.09 % | 41.99 % | 38.43 % | 11.02 % | 6.45 Tbyte/s | 256 | 18 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 64.22 us | 81.47 % | 81.47 % | 41.78 % | 37.25 % | 23.23 % | 6.24 Tbyte/s | 128 | 51 register/thread | 0 byte/block | 0 byte/block | 9 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.03× faster** (33.5 µs vs 34.3 µs).
- **bf16**: Triton is **1.01× faster** (34.6 µs vs 35.0 µs).
- **fp32**: Triton is **1.03× faster** (62.2 µs vs 64.2 µs).

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
