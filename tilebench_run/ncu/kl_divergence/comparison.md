# NCU Comparison: kl_divergence

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'rows': 4096, 'cols': 16384}` | `(default)` | `(default)` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 136.86 us | 46.79 % | 46.79 % | 32.63 % | 25.76 % | 65.67 % | 3.59 Tbyte/s | 1024 | 31 register/thread | 0 byte/block | 128 byte/block | 2 block / 7 block |
| fp32 | cutile | 136.13 us | 47.05 % | 47.05 % | 32.72 % | 25.84 % | 65.74 % | 3.61 Tbyte/s | 1024 | 31 register/thread | 0 byte/block | 128 byte/block | 2 block / 7 block |

## Key findings (auto-derived)

- **fp32**: cuTile is **1.01× faster** (136.1 µs vs 136.9 µs).

## NCU's own bottleneck verdict

- **fp32 / cutile** — Compute is more heavily utilized than Memory
- **fp32 / triton** — Compute is more heavily utilized than Memory

## Reports

- `cutile_fp32.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
