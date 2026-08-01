# NCU Comparison: jacobi_stencil_2d

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'rows': 10240}` | `{'BLOCK_SIZE_R': 1, 'BLOCK_SIZE_C': 1024, 'num_warps': 8}` | `{'tile_r': 4, 'tile_c': 512, 'occupancy': 4}` |
| bf16 | `{'rows': 10240}` | `{'BLOCK_SIZE_R': 1, 'BLOCK_SIZE_C': 1024, 'num_warps': 8}` | `{'tile_r': 4, 'tile_c': 512, 'occupancy': 4}` |
| fp32 | `{'rows': 10240}` | `{'BLOCK_SIZE_R': 1, 'BLOCK_SIZE_C': 256, 'num_warps': 4}` | `{'tile_r': 4, 'tile_c': 512, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 155.58 us | 76.31 % | 32.06 % | 78.64 % | 21.86 % | 56.67 % | 2.46 Tbyte/s | 256 | 30 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp16 | cutile | 134.91 us | 59.58 % | 36.73 % | 61.76 % | 24.17 % | 76.96 % | 2.82 Tbyte/s | 128 | 62 register/thread | 8.20 Kbyte/block | 0 byte/block | 8 block / 14 block |
| bf16 | triton | 155.58 us | 76.37 % | 32.04 % | 78.56 % | 21.83 % | 56.71 % | 2.46 Tbyte/s | 256 | 30 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| bf16 | cutile | 134.34 us | 59.54 % | 36.90 % | 61.80 % | 24.29 % | 76.92 % | 2.83 Tbyte/s | 128 | 62 register/thread | 8.20 Kbyte/block | 0 byte/block | 8 block / 14 block |
| fp32 | triton | 216.67 us | 64.54 % | 48.12 % | 65.73 % | 27.86 % | 51.12 % | 3.69 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| fp32 | cutile | 179.49 us | 80.60 % | 58.09 % | 82.67 % | 34.02 % | 73.12 % | 4.46 Tbyte/s | 128 | 60 register/thread | 16.40 Kbyte/block | 0 byte/block | 8 block / 9 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.15× faster** (134.9 µs vs 155.6 µs).
- **bf16**: cuTile is **1.16× faster** (134.3 µs vs 155.6 µs).
- **fp32**: cuTile is **1.21× faster** (179.5 µs vs 216.7 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Compute is more heavily utilized than Memory
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Compute is more heavily utilized than Memory
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
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
