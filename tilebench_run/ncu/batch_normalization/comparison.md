# NCU Comparison: batch_normalization

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'N': 20000, 'C': 1024, 'eps': 1e-05}` | `{'BLOCK': 1024, 'num_warps': 4}` | `{'tile': 1024, 'occupancy': 16}` |
| bf16 | `{'N': 20000, 'C': 1024, 'eps': 1e-05}` | `{'BLOCK': 1024, 'num_warps': 4}` | `{'tile': 1024, 'occupancy': 16}` |
| fp32 | `{'N': 20000, 'C': 1024, 'eps': 1e-05}` | `{'BLOCK': 1024, 'num_warps': 4}` | `{'tile': 512, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 113.31 us | 92.57 % | 6.16 % | 97.20 % | 64.42 % | 12.77 % | 472.55 Gbyte/s | 128 | 26 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp16 | cutile | 140.32 us | 91.96 % | 6.12 % | 97.27 % | 64.19 % | 14.49 % | 469.65 Gbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| bf16 | triton | 113.47 us | 91.78 % | 6.14 % | 97.39 % | 64.27 % | 12.67 % | 470.69 Gbyte/s | 128 | 26 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| bf16 | cutile | 139.65 us | 92.19 % | 6.13 % | 97.42 % | 64.05 % | 14.52 % | 470.32 Gbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 117.44 us | 92.52 % | 12.85 % | 98.39 % | 67.33 % | 11.94 % | 985.77 Gbyte/s | 128 | 25 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp32 | cutile | 142.56 us | 92.78 % | 12.83 % | 98.25 % | 67.40 % | 13.71 % | 983.82 Gbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.24× faster** (113.3 µs vs 140.3 µs).
- **bf16**: Triton is **1.23× faster** (113.5 µs vs 139.7 µs).
- **fp32**: Triton is **1.21× faster** (117.4 µs vs 142.6 µs).

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
