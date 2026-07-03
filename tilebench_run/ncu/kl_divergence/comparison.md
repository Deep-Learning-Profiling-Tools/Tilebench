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
| fp32 | triton | 95.01 us | 74.13 % | 74.13 % | 18.14 % | 49.06 % | 71.20 % | 5.69 Tbyte/s | 128 | 64 register/thread | 0 byte/block | 16 byte/block | 8 block / 28 block |
| fp32 | cutile | 88.83 us | 79.35 % | 79.35 % | 22.37 % | 52.80 % | 71.50 % | 6.08 Tbyte/s | 256 | 64 register/thread | 49.25 Kbyte/block | 0 byte/block | 4 block / 4 block |

## Key findings (auto-derived)

- **fp32**: cuTile is **1.07× faster** (88.8 µs vs 95.0 µs).

## NCU's own bottleneck verdict

- **fp32 / cutile** — Compute and Memory are well-balanced
- **fp32 / triton** — Compute and Memory are well-balanced

## Reports

- `cutile_fp32.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
