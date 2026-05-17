# NCU Comparison: radix_sort

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| int32 | `{'n': 20000000}` | `{'num_warps': 4}` | `{'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| int32 | triton | 4.54 us | 2.54 % | 0.24 % | 7.54 % | 0.29 % | 0.10 % | 18.03 Gbyte/s | 128 | 16 register/thread | 0 byte/block | 16 byte/block | 32 block / 28 block |
| int32 | cutile | 4.51 us | 2.58 % | 0.24 % | 3.05 % | 0.29 % | 0.14 % | 18.21 Gbyte/s | 128 | 26 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |

## Key findings (auto-derived)

- **int32**: cuTile is **1.01× faster** (4.5 µs vs 4.5 µs).

## NCU's own bottleneck verdict

- **int32 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.01 full waves across all SMs. Look at Launch Statistics for more details.
- **int32 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.01 full waves across all SMs. Look at Launch Statistics for more details.

## Reports

- `cutile_int32.ncu-rep`
- `triton_int32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
