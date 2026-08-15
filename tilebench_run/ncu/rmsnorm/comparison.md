# NCU Comparison: rmsnorm

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'M': 2048, 'K': 10240}` | `{'BLOCK_N_SIZE': 2048, 'num_warps': 4, 'num_stages': 2}` | `{'tile_size': 1024, 'occupancy': 16}` |
| bf16 | `{'batch': 1, 'M': 2048, 'K': 10240}` | `{'BLOCK_N_SIZE': 2048, 'num_warps': 4, 'num_stages': 2}` | `{'tile_size': 2048, 'occupancy': 16}` |
| fp32 | `{'batch': 1, 'M': 2048, 'K': 10240}` | `{'BLOCK_N_SIZE': 2048, 'num_warps': 8, 'num_stages': 2}` | `{'tile_size': 1024, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 21.79 us | 54.96 % | 54.96 % | 34.80 % | 28.12 % | 29.36 % | 4.21 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp16 | cutile | 23.84 us | 50.57 % | 50.57 % | 29.63 % | 25.88 % | 41.36 % | 3.87 Tbyte/s | 128 | 31 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| bf16 | triton | 21.73 us | 54.75 % | 54.75 % | 34.32 % | 28.10 % | 31.37 % | 4.19 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| bf16 | cutile | 23.10 us | 52.16 % | 52.16 % | 31.08 % | 26.59 % | 40.28 % | 3.99 Tbyte/s | 128 | 32 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 35.58 us | 67.72 % | 67.72 % | 38.93 % | 34.88 % | 19.66 % | 5.19 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp32 | cutile | 36.10 us | 64.62 % | 64.62 % | 36.06 % | 33.53 % | 23.72 % | 4.95 Tbyte/s | 128 | 68 register/thread | 28 byte/block | 0 byte/block | 7 block / 14 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.09× faster** (21.8 µs vs 23.8 µs).
- **bf16**: Triton is **1.06× faster** (21.7 µs vs 23.1 µs).
- **fp32**: Triton is **1.01× faster** (35.6 µs vs 36.1 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **bf16 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
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
