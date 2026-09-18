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
| fp16 | triton | 163.20 us | 72.61 % | 35.13 % | 74.28 % | 21.80 % | 54.07 % | 2.69 Tbyte/s | 256 | 30 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp16 | cutile | 136.32 us | 59.09 % | 41.75 % | 60.75 % | 23.96 % | 76.43 % | 3.20 Tbyte/s | 128 | 62 register/thread | 8.20 Kbyte/block | 0 byte/block | 8 block / 14 block |
| bf16 | triton | 163.65 us | 72.25 % | 35.02 % | 74.23 % | 21.76 % | 53.81 % | 2.69 Tbyte/s | 256 | 30 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| bf16 | cutile | 135.87 us | 59.09 % | 41.92 % | 60.70 % | 24.04 % | 76.42 % | 3.22 Tbyte/s | 128 | 62 register/thread | 8.20 Kbyte/block | 0 byte/block | 8 block / 14 block |
| fp32 | triton | 222.50 us | 62.80 % | 50.17 % | 63.51 % | 28.00 % | 49.86 % | 3.85 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| fp32 | cutile | 185.18 us | 78.26 % | 60.26 % | 79.88 % | 33.79 % | 71.10 % | 4.62 Tbyte/s | 128 | 60 register/thread | 16.40 Kbyte/block | 0 byte/block | 8 block / 9 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.20× faster** (136.3 µs vs 163.2 µs).
- **bf16**: cuTile is **1.20× faster** (135.9 µs vs 163.7 µs).
- **fp32**: cuTile is **1.20× faster** (185.2 µs vs 222.5 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Compute is more heavily utilized than Memory
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Compute is more heavily utilized than Memory
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — Compute and Memory are well-balanced
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
