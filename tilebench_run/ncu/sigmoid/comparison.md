# NCU Comparison: sigmoid

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 50000000}` | `{'BLOCK_SIZE': 4096, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 4096, 'occupancy': 16}` |
| bf16 | `{'n': 50000000}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 4096, 'occupancy': 16}` |
| fp32 | `{'n': 50000000}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 4096, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 30.69 us | 63.30 % | 63.30 % | 47.26 % | 35.47 % | 72.62 % | 4.85 Tbyte/s | 256 | 29 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp16 | cutile | 50.98 us | 39.99 % | 39.99 % | 27.18 % | 21.56 % | 73.51 % | 3.06 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | triton | 30.53 us | 64.32 % | 64.32 % | 46.58 % | 35.59 % | 73.36 % | 4.93 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | cutile | 50.78 us | 40.14 % | 40.14 % | 27.12 % | 21.62 % | 74.10 % | 3.08 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | triton | 56.86 us | 79.71 % | 79.71 % | 48.19 % | 39.99 % | 39.12 % | 6.11 Tbyte/s | 256 | 21 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 65.28 us | 70.26 % | 70.26 % | 41.70 % | 35.01 % | 56.77 % | 5.39 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.66× faster** (30.7 µs vs 51.0 µs).
- **bf16**: Triton is **1.66× faster** (30.5 µs vs 50.8 µs).
- **fp32**: Triton is **1.15× faster** (56.9 µs vs 65.3 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Compute is more heavily utilized than Memory
- **bf16 / triton** — Compute and Memory are well-balanced
- **fp16 / cutile** — Compute is more heavily utilized than Memory
- **fp16 / triton** — Compute and Memory are well-balanced
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
