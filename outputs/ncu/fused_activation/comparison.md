# NCU Comparison: fused_activation

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'dtype': 'fp32', 'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'tile': 1024, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 53.79 us | 85.01 % | 85.01 % | 24.82 % | 40.68 % | 19.13 % | 6.52 Tbyte/s | 64 | 35 register/thread | 0 byte/block | 0 byte/block | 24 block / 32 block |
| fp32 | cutile | 54.62 us | 83.65 % | 83.65 % | 25.38 % | 40.05 % | 46.21 % | 6.41 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |

## Key findings (auto-derived)

- **fp32**: Triton is **1.02× faster** (53.8 µs vs 54.6 µs).

## NCU's own bottleneck verdict

- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.

## Reports

- `cutile_fp32.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
