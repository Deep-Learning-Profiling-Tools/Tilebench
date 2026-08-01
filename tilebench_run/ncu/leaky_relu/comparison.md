# NCU Comparison: leaky_relu

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 50000000}` | `{'BLOCK_SIZE': 4096, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 4096, 'occupancy': 4}` |
| bf16 | `{'n': 50000000}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 2048, 'occupancy': 16}` |
| fp32 | `{'n': 50000000}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 1024, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 26.88 us | 69.92 % | 69.92 % | 55.47 % | 40.35 % | 27.57 % | 5.35 Tbyte/s | 256 | 24 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp16 | cutile | 27.14 us | 69.21 % | 69.21 % | 56.31 % | 39.80 % | 49.37 % | 5.30 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | triton | 32.00 us | 63.54 % | 63.54 % | 44.27 % | 34.33 % | 23.10 % | 4.87 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| bf16 | cutile | 27.04 us | 69.70 % | 69.70 % | 55.23 % | 40.17 % | 53.09 % | 5.34 Tbyte/s | 128 | 22 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| fp32 | triton | 56.74 us | 79.90 % | 79.90 % | 47.73 % | 40.09 % | 12.53 % | 6.13 Tbyte/s | 256 | 18 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 56.16 us | 80.34 % | 80.34 % | 48.33 % | 40.39 % | 33.29 % | 6.16 Tbyte/s | 128 | 19 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.01× faster** (26.9 µs vs 27.1 µs).
- **bf16**: cuTile is **1.18× faster** (27.0 µs vs 32.0 µs).
- **fp32**: cuTile is **1.01× faster** (56.2 µs vs 56.7 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Memory is more heavily utilized than Compute
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Memory is more heavily utilized than Compute
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **fp32 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
