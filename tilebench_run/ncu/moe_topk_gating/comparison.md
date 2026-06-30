# NCU Comparison: moe_topk_gating

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'E': 128, 'k': 2, 'M': 20480}` | `{'num_warps': 1}` | `{'occupancy': 32}` |
| bf16 | `{'E': 128, 'k': 2, 'M': 20480}` | `{'num_warps': 1}` | `{'occupancy': 32}` |
| fp32 | `{'E': 128, 'k': 2, 'M': 20480}` | `{'num_warps': 1}` | `{'occupancy': 32}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 25.34 us | 69.88 % | 2.71 % | 83.46 % | 2.20 % | 64.66 % | 207.12 Gbyte/s | 128 | 22 register/thread | 0 byte/block | 24 byte/block | 21 block / 28 block |
| fp16 | cutile | 29.57 us | 64.24 % | 2.32 % | 74.78 % | 2.35 % | 68.00 % | 177.60 Gbyte/s | 128 | 22 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |
| bf16 | triton | 24.90 us | 71.01 % | 2.75 % | 84.80 % | 2.22 % | 63.71 % | 210.84 Gbyte/s | 128 | 18 register/thread | 0 byte/block | 32 byte/block | 21 block / 28 block |
| bf16 | cutile | 29.47 us | 64.43 % | 2.33 % | 74.17 % | 2.35 % | 68.19 % | 178.19 Gbyte/s | 128 | 22 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |
| fp32 | triton | 25.22 us | 69.18 % | 5.43 % | 84.42 % | 3.61 % | 62.01 % | 416.06 Gbyte/s | 128 | 20 register/thread | 0 byte/block | 32 byte/block | 21 block / 28 block |
| fp32 | cutile | 29.44 us | 64.57 % | 4.65 % | 75.08 % | 3.09 % | 67.81 % | 356.46 Gbyte/s | 128 | 22 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.17× faster** (25.3 µs vs 29.6 µs).
- **bf16**: Triton is **1.18× faster** (24.9 µs vs 29.5 µs).
- **fp32**: Triton is **1.17× faster** (25.2 µs vs 29.4 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Compute and Memory are well-balanced
- **bf16 / triton** — Compute and Memory are well-balanced
- **fp16 / cutile** — Compute and Memory are well-balanced
- **fp16 / triton** — Compute and Memory are well-balanced
- **fp32 / cutile** — Compute and Memory are well-balanced
- **fp32 / triton** — Compute and Memory are well-balanced

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
