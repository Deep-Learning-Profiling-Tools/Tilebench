# NCU Comparison: gaussian_blur

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'kernel_rows': 7, 'kernel_cols': 7, 'input_rows': 10240}` | `{'BLOCK_R': 8, 'BLOCK_C': 64, 'num_warps': 8}` | `{'tile_r': 2, 'tile_c': 128, 'occupancy': 4}` |
| fp32 | `{'kernel_rows': 7, 'kernel_cols': 7, 'input_rows': 10240}` | `{'BLOCK_R': 8, 'BLOCK_C': 64, 'num_warps': 8}` | `{'tile_r': 2, 'tile_c': 128, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 1030.00 us | 76.48 % | 4.87 % | 76.82 % | 4.04 % | 85.47 % | 373.35 Gbyte/s | 256 | 34 register/thread | 0 byte/block | 0 byte/block | 6 block / 32 block |
| fp16 | cutile | 1510.00 us | 60.52 % | 3.34 % | 60.71 % | 5.77 % | 84.44 % | 256.01 Gbyte/s | 128 | 48 register/thread | 524 byte/block | 0 byte/block | 10 block / 39 block |
| fp32 | triton | 1280.00 us | 98.55 % | 8.20 % | 98.96 % | 5.31 % | 57.66 % | 628.87 Gbyte/s | 256 | 32 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp32 | cutile | 1290.00 us | 76.36 % | 8.15 % | 76.66 % | 8.84 % | 77.52 % | 625.04 Gbyte/s | 128 | 48 register/thread | 1.04 Kbyte/block | 0 byte/block | 10 block / 30 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.47× faster** (1030.0 µs vs 1510.0 µs).
- **fp32**: Triton is **1.01× faster** (1280.0 µs vs 1290.0 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp16 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp32 / cutile** — Compute and Memory are well-balanced
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
