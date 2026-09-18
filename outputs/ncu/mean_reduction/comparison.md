# NCU Comparison: mean_reduction

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_M': 1, 'BLOCK_N': 1024, 'num_warps': 8}` | `{'tile_size': 512, 'occupancy': 4}` |
| bf16 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_M': 1, 'BLOCK_N': 2048, 'num_warps': 8}` | `{'tile_size': 1024, 'occupancy': 16}` |
| fp32 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_M': 1, 'BLOCK_N': 512, 'num_warps': 8}` | `{'tile_size': 1024, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 21.15 us | 68.59 % | 68.59 % | 18.65 % | 34.63 % | 17.59 % | 5.25 Tbyte/s | 256 | 31 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp16 | cutile | 22.02 us | 63.07 % | 63.07 % | 19.81 % | 33.35 % | 34.34 % | 4.83 Tbyte/s | 128 | 74 register/thread | 28 byte/block | 0 byte/block | 6 block / 14 block |
| bf16 | triton | 19.97 us | 68.34 % | 68.34 % | 18.86 % | 36.71 % | 16.71 % | 5.24 Tbyte/s | 256 | 30 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| bf16 | cutile | 25.15 us | 60.87 % | 60.87 % | 13.30 % | 29.04 % | 23.22 % | 4.66 Tbyte/s | 128 | 28 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 37.22 us | 77.15 % | 77.15 % | 16.08 % | 39.27 % | 8.45 % | 5.91 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp32 | cutile | 39.04 us | 74.71 % | 74.71 % | 14.59 % | 37.44 % | 17.06 % | 5.73 Tbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.04× faster** (21.1 µs vs 22.0 µs).
- **bf16**: Triton is **1.26× faster** (20.0 µs vs 25.1 µs).
- **fp32**: Triton is **1.05× faster** (37.2 µs vs 39.0 µs).

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
