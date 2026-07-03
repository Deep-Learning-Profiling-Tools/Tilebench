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
| fp16 | triton | 1790.00 us | 53.27 % | 2.84 % | 53.40 % | 4.28 % | 82.88 % | 218.06 Gbyte/s | 256 | 62 register/thread | 0 byte/block | 0 byte/block | 4 block / 32 block |
| fp16 | cutile | 4600.00 us | 16.11 % | 1.10 % | 16.13 % | 2.29 % | 92.80 % | 84.64 Gbyte/s | 128 | 54 register/thread | 1.04 Kbyte/block | 0 byte/block | 9 block / 30 block |
| fp32 | triton | 1610.00 us | 93.35 % | 6.53 % | 93.63 % | 8.04 % | 81.94 % | 500.62 Gbyte/s | 256 | 32 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp32 | cutile | 5310.00 us | 14.45 % | 1.98 % | 14.47 % | 2.57 % | 94.80 % | 152.12 Gbyte/s | 128 | 56 register/thread | 2.06 Kbyte/block | 0 byte/block | 9 block / 20 block |

## Key findings (auto-derived)

- **fp16**: Triton is **2.57× faster** (1790.0 µs vs 4600.0 µs).
- **fp32**: Triton is **3.30× faster** (1610.0 µs vs 5310.0 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp16 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
