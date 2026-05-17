# NCU Comparison: 2d_max_pooling

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'N': 4, 'C': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'H': 640}` | `{'BLOCK_SIZE': 1024, 'num_warps': 8}` | `{'tile': 256, 'occupancy': 4}` |
| bf16 | `{'N': 4, 'C': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'H': 640}` | `{'BLOCK_SIZE': 1024, 'num_warps': 8}` | `{'tile': 256, 'occupancy': 4}` |
| fp32 | `{'N': 4, 'C': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'H': 640}` | `{'BLOCK_SIZE': 256, 'num_warps': 4}` | `{'tile': 256, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 229.15 us | 97.48 % | 29.00 % | 99.36 % | 15.84 % | 80.93 % | 2.22 Tbyte/s | 256 | 48 register/thread | 0 byte/block | 0 byte/block | 5 block / 32 block |
| fp16 | cutile | 598.43 us | 11.98 % | 11.14 % | 12.07 % | 6.61 % | 87.68 % | 854.55 Gbyte/s | 128 | 38 register/thread | 524 byte/block | 0 byte/block | 12 block / 39 block |
| bf16 | triton | 228.86 us | 97.47 % | 29.02 % | 99.34 % | 15.86 % | 81.22 % | 2.23 Tbyte/s | 256 | 48 register/thread | 0 byte/block | 0 byte/block | 5 block / 32 block |
| bf16 | cutile | 578.56 us | 12.26 % | 11.52 % | 12.36 % | 6.84 % | 87.26 % | 883.58 Gbyte/s | 128 | 36 register/thread | 524 byte/block | 0 byte/block | 12 block / 39 block |
| fp32 | triton | 265.50 us | 88.21 % | 50.77 % | 89.56 % | 28.44 % | 86.46 % | 3.89 Tbyte/s | 128 | 31 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | cutile | 592.61 us | 22.76 % | 22.76 % | 22.12 % | 13.16 % | 87.30 % | 1.75 Tbyte/s | 128 | 37 register/thread | 1.04 Kbyte/block | 0 byte/block | 12 block / 30 block |

## Key findings (auto-derived)

- **fp16**: Triton is **2.61× faster** (229.2 µs vs 598.4 µs).
- **bf16**: Triton is **2.53× faster** (228.9 µs vs 578.6 µs).
- **fp32**: Triton is **2.23× faster** (265.5 µs vs 592.6 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **bf16 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp16 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
