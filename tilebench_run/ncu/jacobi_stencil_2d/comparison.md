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
| fp16 | triton | 155.49 us | 76.70 % | 32.11 % | 78.70 % | 22.07 % | 56.96 % | 2.46 Tbyte/s | 256 | 30 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp16 | cutile | 134.69 us | 59.91 % | 36.80 % | 61.79 % | 24.26 % | 77.40 % | 2.82 Tbyte/s | 128 | 62 register/thread | 8.20 Kbyte/block | 0 byte/block | 8 block / 14 block |
| bf16 | triton | 155.33 us | 76.46 % | 32.16 % | 78.51 % | 22.09 % | 56.77 % | 2.47 Tbyte/s | 256 | 30 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| bf16 | cutile | 134.53 us | 59.57 % | 36.85 % | 61.78 % | 24.27 % | 76.92 % | 2.83 Tbyte/s | 128 | 62 register/thread | 8.20 Kbyte/block | 0 byte/block | 8 block / 14 block |
| fp32 | triton | 217.06 us | 64.54 % | 48.06 % | 65.77 % | 27.98 % | 51.10 % | 3.69 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| fp32 | cutile | 197.79 us | 70.22 % | 52.70 % | 72.02 % | 29.16 % | 59.89 % | 4.04 Tbyte/s | 128 | 80 register/thread | 16.40 Kbyte/block | 0 byte/block | 6 block / 7 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.15× faster** (134.7 µs vs 155.5 µs).
- **bf16**: cuTile is **1.15× faster** (134.5 µs vs 155.3 µs).
- **fp32**: cuTile is **1.10× faster** (197.8 µs vs 217.1 µs).

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
