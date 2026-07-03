# NCU Comparison: jacobi_stencil_2d

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'rows': 10240}` | `{'BLOCK_SIZE_R': 1, 'BLOCK_SIZE_C': 1024, 'num_warps': 8}` | `{'tile_r': 4, 'tile_c': 512, 'occupancy': 4}` |
| bf16 | `{'rows': 10240}` | `{'BLOCK_SIZE_R': 1, 'BLOCK_SIZE_C': 1024, 'num_warps': 8}` | `{'tile_r': 4, 'tile_c': 512, 'occupancy': 4}` |
| fp32 | `{'rows': 10240}` | `{'BLOCK_SIZE_R': 1, 'BLOCK_SIZE_C': 256, 'num_warps': 4}` | `{'tile_r': 4, 'tile_c': 512, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 155.26 us | 76.52 % | 32.17 % | 78.84 % | 21.89 % | 56.81 % | 2.47 Tbyte/s | 256 | 30 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp16 | cutile | 134.98 us | 59.60 % | 36.71 % | 61.79 % | 24.18 % | 77.01 % | 2.82 Tbyte/s | 128 | 62 register/thread | 8.20 Kbyte/block | 0 byte/block | 8 block / 14 block |
| bf16 | triton | 155.42 us | 76.57 % | 32.10 % | 78.69 % | 21.86 % | 56.85 % | 2.46 Tbyte/s | 256 | 30 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| bf16 | cutile | 134.50 us | 59.63 % | 36.84 % | 61.83 % | 24.27 % | 77.01 % | 2.83 Tbyte/s | 128 | 62 register/thread | 8.20 Kbyte/block | 0 byte/block | 8 block / 14 block |
| fp32 | triton | 217.38 us | 64.40 % | 47.96 % | 65.72 % | 27.79 % | 51.00 % | 3.68 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| fp32 | cutile | 198.27 us | 70.31 % | 52.57 % | 72.19 % | 29.02 % | 59.97 % | 4.03 Tbyte/s | 128 | 80 register/thread | 16.40 Kbyte/block | 0 byte/block | 6 block / 7 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.15× faster** (135.0 µs vs 155.3 µs).
- **bf16**: cuTile is **1.16× faster** (134.5 µs vs 155.4 µs).
- **fp32**: cuTile is **1.10× faster** (198.3 µs vs 217.4 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Compute is more heavily utilized than Memory
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Compute is more heavily utilized than Memory
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — Memory is more heavily utilized than Compute
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
