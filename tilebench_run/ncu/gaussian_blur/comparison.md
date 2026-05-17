# NCU Comparison: gaussian_blur

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'kernel_rows': 7, 'kernel_cols': 7, 'input_rows': 10240}` | `{'BLOCK_SIZE': 512, 'num_warps': 8}` | `{'tile': 512, 'occupancy': 4}` |
| fp32 | `{'kernel_rows': 7, 'kernel_cols': 7, 'input_rows': 10240}` | `{'BLOCK_SIZE': 512, 'num_warps': 8}` | `{'tile': 512, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 4.77 us | 2.45 % | 0.08 % | 49.64 % | 0.32 % | 0.00 % | 6.44 Gbyte/s | 128 | 20 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| fp16 | cutile | 4.61 us | 2.54 % | 0.09 % | 50.02 % | 0.33 % | 0.00 % | 6.67 Gbyte/s | 128 | 20 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| fp32 | triton | 4.67 us | 2.52 % | 0.05 % | 43.21 % | 0.27 % | 0.00 % | 3.78 Gbyte/s | 128 | 20 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| fp32 | cutile | 4.54 us | 2.58 % | 0.05 % | 44.71 % | 0.28 % | 0.00 % | 3.89 Gbyte/s | 128 | 20 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.03× faster** (4.6 µs vs 4.8 µs).
- **fp32**: cuTile is **1.03× faster** (4.5 µs vs 4.7 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.00 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.00 full waves across all SMs. Look at Launch Statistics for more details.
- **fp32 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.00 full waves across all SMs. Look at Launch Statistics for more details.
- **fp32 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.00 full waves across all SMs. Look at Launch Statistics for more details.

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
