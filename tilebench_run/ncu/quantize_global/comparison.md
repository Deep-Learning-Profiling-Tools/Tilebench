# NCU Comparison: quantize_global

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'dtype': 'fp32', 'n': 20971520}` | `{'BLOCK_SIZE': 4096, 'num_warps': 4}` | `{'tile': 4096, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 17.70 us | 68.76 % | 68.76 % | 49.81 % | 41.51 % | 7.98 % | 5.26 Tbyte/s | 128 | 34 register/thread | 0 byte/block | 0 byte/block | 12 block / 32 block |
| fp32 | cutile | 18.46 us | 66.62 % | 66.62 % | 57.93 % | 39.90 % | 26.56 % | 5.10 Tbyte/s | 128 | 43 register/thread | 16.40 Kbyte/block | 0 byte/block | 10 block / 11 block |

## Key findings (auto-derived)

- **fp32**: Triton is **1.04× faster** (17.7 µs vs 18.5 µs).

## NCU's own bottleneck verdict

- **fp32 / cutile** — Memory is more heavily utilized than Compute
- **fp32 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_fp32.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
