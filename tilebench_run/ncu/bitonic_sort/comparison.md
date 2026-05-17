# NCU Comparison: bitonic_sort

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 10000000}` | `{'BLOCK': 512, 'num_warps': 4}` | `{'tile': 1024, 'occupancy': 4}` |
| fp32 | `{'n': 10000000}` | `{'BLOCK': 512, 'num_warps': 4}` | `{'tile': 1024, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 27.62 us | 59.45 % | 15.89 % | 70.07 % | 39.93 % | 77.44 % | 1.22 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 1.02 Kbyte/block | 16 block / 32 block |
| fp16 | cutile | 29.60 us | 56.01 % | 14.82 % | 64.85 % | 24.68 % | 63.04 % | 1.14 Tbyte/s | 128 | 40 register/thread | 6.16 Kbyte/block | 0 byte/block | 12 block / 18 block |
| fp32 | triton | 31.01 us | 76.02 % | 33.89 % | 87.45 % | 47.85 % | 59.63 % | 2.60 Tbyte/s | 128 | 28 register/thread | 0 byte/block | 2.05 Kbyte/block | 16 block / 33 block |
| fp32 | cutile | 34.75 us | 62.45 % | 30.22 % | 69.67 % | 37.85 % | 59.99 % | 2.32 Tbyte/s | 128 | 53 register/thread | 12.30 Kbyte/block | 0 byte/block | 9 block / 10 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.07× faster** (27.6 µs vs 29.6 µs).
- **fp32**: Triton is **1.12× faster** (31.0 µs vs 34.8 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — Compute and Memory are well-balanced
- **fp16 / triton** — Compute is more heavily utilized than Memory
- **fp32 / cutile** — Compute and Memory are well-balanced
- **fp32 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
