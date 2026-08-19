# NCU Comparison: dropout

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'p': 0.5, 'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'tile': 2048, 'occupancy': 16}` |
| bf16 | `{'p': 0.5, 'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'tile': 2048, 'occupancy': 16}` |
| fp32 | `{'p': 0.5, 'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'tile': 512, 'occupancy': 32}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 23.68 us | 71.50 % | 71.50 % | 29.21 % | 33.24 % | 22.89 % | 5.47 Tbyte/s | 64 | 32 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| fp16 | cutile | 24.29 us | 69.21 % | 69.21 % | 28.33 % | 32.40 % | 24.98 % | 5.30 Tbyte/s | 128 | 26 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | triton | 23.52 us | 72.38 % | 72.38 % | 29.17 % | 33.52 % | 23.66 % | 5.54 Tbyte/s | 64 | 32 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | cutile | 23.97 us | 71.06 % | 71.06 % | 28.10 % | 32.87 % | 27.15 % | 5.45 Tbyte/s | 128 | 27 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | triton | 42.75 us | 79.04 % | 79.04 % | 29.98 % | 36.60 % | 11.47 % | 6.06 Tbyte/s | 64 | 34 register/thread | 0 byte/block | 0 byte/block | 24 block / 32 block |
| fp32 | cutile | 42.85 us | 79.82 % | 79.82 % | 29.31 % | 36.70 % | 30.61 % | 6.11 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.03× faster** (23.7 µs vs 24.3 µs).
- **bf16**: Triton is **1.02× faster** (23.5 µs vs 24.0 µs).
- **fp32**: Triton is **1.00× faster** (42.8 µs vs 42.9 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Memory is more heavily utilized than Compute
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Memory is more heavily utilized than Compute
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
