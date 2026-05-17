# NCU Comparison: batch_normalization

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'C': 1024, 'eps': 1e-05, 'N': 20000}` | `{'BLOCK': 1024, 'num_warps': 4}` | `{'tile': 1024, 'occupancy': 16}` |
| bf16 | `{'C': 1024, 'eps': 1e-05, 'N': 20000}` | `{'BLOCK': 1024, 'num_warps': 4}` | `{'tile': 1024, 'occupancy': 16}` |
| fp32 | `{'C': 1024, 'eps': 1e-05, 'N': 20000}` | `{'BLOCK': 1024, 'num_warps': 4}` | `{'tile': 1024, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 86.94 us | 92.36 % | 6.15 % | 97.50 % | 64.39 % | 12.75 % | 471.52 Gbyte/s | 128 | 26 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp16 | cutile | 87.04 us | 92.06 % | 6.14 % | 97.32 % | 64.24 % | 14.50 % | 470.94 Gbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| bf16 | triton | 87.23 us | 92.40 % | 6.14 % | 97.44 % | 64.23 % | 12.75 % | 470.56 Gbyte/s | 128 | 26 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| bf16 | cutile | 87.07 us | 92.27 % | 6.15 % | 97.30 % | 64.15 % | 14.54 % | 471.47 Gbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 86.21 us | 93.10 % | 12.85 % | 98.46 % | 67.59 % | 12.02 % | 984.97 Gbyte/s | 128 | 25 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp32 | cutile | 86.14 us | 92.71 % | 12.85 % | 98.11 % | 67.62 % | 13.70 % | 985.64 Gbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.00× faster** (86.9 µs vs 87.0 µs).
- **bf16**: cuTile is **1.00× faster** (87.1 µs vs 87.2 µs).
- **fp32**: cuTile is **1.00× faster** (86.1 µs vs 86.2 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **bf16 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **fp16 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
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
