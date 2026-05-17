# NCU Comparison: 3d_conv

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'input_depth': 64, 'kernel_depth': 3, 'kernel_rows': 3, 'kernel_cols': 3, 'input_rows': 640}` | `{'BLOCK_SIZE': 512, 'num_warps': 4}` | `{'tile': 512, 'occupancy': 8}` |
| fp32 | `{'input_depth': 64, 'kernel_depth': 3, 'kernel_rows': 3, 'kernel_cols': 3, 'input_rows': 640}` | `{'BLOCK_SIZE': 256, 'num_warps': 4}` | `{'tile': 512, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 21.41 us | 73.38 % | 73.38 % | 39.92 % | 41.22 % | 13.20 % | 5.61 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 251.17 us | 43.41 % | 3.73 % | 44.21 % | 11.65 % | 78.00 % | 285.90 Gbyte/s | 128 | 64 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp32 | triton | 235.84 us | 94.43 % | 9.48 % | 96.34 % | 16.82 % | 76.86 % | 727.04 Gbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | cutile | 216.51 us | 52.16 % | 10.37 % | 53.37 % | 19.59 % | 74.69 % | 795.40 Gbyte/s | 128 | 64 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **11.73× faster** (21.4 µs vs 251.2 µs).
- **fp32**: cuTile is **1.09× faster** (216.5 µs vs 235.8 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — Compute is more heavily utilized than Memory
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — Compute is more heavily utilized than Memory
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
