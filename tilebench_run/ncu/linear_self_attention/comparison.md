# NCU Comparison: linear_self_attention

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| float32 | `{'eps': 1e-06, 'M': 10000, 'D': 256}` | `{'BLOCK_M': 32, 'BLOCK_D': 32, 'num_warps': 1, 'num_stages': 1}` | `{'block_m': 64, 'block_d': 32, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float32 | triton | 4900.00 us | 99.48 % | 0.05 % | 99.93 % | 64.82 % | 11.60 % | 4.18 Gbyte/s | 32 | 29 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| float32 | cutile | 19580.00 us | 99.42 % | 0.01 % | 99.93 % | 50.91 % | 22.46 % | 1.05 Gbyte/s | 128 | 28 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |

## Key findings (auto-derived)

- **float32**: Triton is **4.00× faster** (4900.0 µs vs 19580.0 µs).

## NCU's own bottleneck verdict

- **float32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **float32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.

## Reports

- `cutile_float32.ncu-rep`
- `triton_float32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
