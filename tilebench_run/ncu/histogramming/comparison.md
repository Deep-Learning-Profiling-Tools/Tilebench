# NCU Comparison: histogramming

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| int32 | `{'N': 67108864, 'num_bins': 4096}` | `(default)` | `(default)` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| int32 | triton | 39.81 us | 1.38 % | 1.38 % | 4.80 % | 0.91 % | 0.25 % | 105.58 Gbyte/s | 128 | 48 register/thread | 0 byte/block | 2.05 Kbyte/block | 10 block / 21 block |
| int32 | cutile | 8.06 us | 6.84 % | 6.84 % | 36.67 % | 4.54 % | 0.73 % | 521.78 Gbyte/s | 256 | 255 register/thread | 199.89 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Key findings (auto-derived)

- **int32**: cuTile is **4.94× faster** (8.1 µs vs 39.8 µs).

## NCU's own bottleneck verdict

- **int32 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.11 full waves across all SMs. Look at Launch Statistics for more details.
- **int32 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.01 full waves across all SMs. Look at Launch Statistics for more details.

## Reports

- `cutile_int32.ncu-rep`
- `triton_int32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
