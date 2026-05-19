# NCU Comparison: kl_divergence

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'rows': 4096, 'cols': 16384}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4, 'num_stages': 3}` | `{'tile': 2048, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 136.26 us | 47.00 % | 47.00 % | 32.63 % | 25.98 % | 65.50 % | 3.61 Tbyte/s | 1024 | 31 register/thread | 0 byte/block | 128 byte/block | 2 block / 7 block |
| fp32 | cutile | 136.90 us | 46.80 % | 46.80 % | 32.63 % | 25.84 % | 65.60 % | 3.59 Tbyte/s | 1024 | 31 register/thread | 0 byte/block | 128 byte/block | 2 block / 7 block |

## Key findings (auto-derived)

- **fp32**: Triton is **1.00× faster** (136.3 µs vs 136.9 µs).

## NCU's own bottleneck verdict

- **fp32 / cutile** — Compute is more heavily utilized than Memory
- **fp32 / triton** — Compute is more heavily utilized than Memory

## Reports

- `cutile_fp32.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
