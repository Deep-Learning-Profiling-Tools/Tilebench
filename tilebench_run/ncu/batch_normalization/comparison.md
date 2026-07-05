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
| fp16 | triton | 113.66 us | 92.01 % | 6.15 % | 97.42 % | 64.40 % | 12.70 % | 471.87 Gbyte/s | 128 | 26 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp16 | cutile | 140.16 us | 91.83 % | 6.15 % | 97.28 % | 64.41 % | 14.47 % | 471.73 Gbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| bf16 | triton | 113.40 us | 91.91 % | 6.18 % | 97.29 % | 64.67 % | 12.68 % | 473.97 Gbyte/s | 128 | 26 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| bf16 | cutile | 139.14 us | 92.28 % | 6.16 % | 97.39 % | 64.33 % | 14.54 % | 472.08 Gbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 117.42 us | 92.50 % | 12.90 % | 98.37 % | 67.80 % | 11.94 % | 989.28 Gbyte/s | 128 | 25 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp32 | cutile | 141.25 us | 93.35 % | 12.92 % | 98.27 % | 67.76 % | 13.79 % | 990.82 Gbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| bf16 | cutile | 1/3 | 86.98 us | `compute_block_sums_kernel_Kt1_A1bf16_1i16t1_p16_A1` |
| bf16 | cutile | 2/3 | 5.47 us | `compute_mean_invstd_kernel_Kt1_A1f32_1i16t1_p16_A1` |
| bf16 | cutile | 3/3 | 46.69 us | `apply_batch_norm_kernel_Kt1_A1bf16_1i16t1_p16_A1bf` |
| bf16 | triton | 1/3 | 86.62 us | `compute_block_sums_kernel` |
| bf16 | triton | 2/3 | 5.66 us | `compute_mean_invstd_kernel` |
| bf16 | triton | 3/3 | 21.12 us | `apply_batch_norm_kernel` |
| fp16 | cutile | 1/3 | 87.04 us | `compute_block_sums_kernel_Kt1_A1f16_1i16t1_p16_A1f` |
| fp16 | cutile | 2/3 | 5.76 us | `compute_mean_invstd_kernel_Kt1_A1f32_1i16t1_p16_A1` |
| fp16 | cutile | 3/3 | 47.36 us | `apply_batch_norm_kernel_Kt1_A1f16_1i16t1_p16_A1f16` |
| fp16 | triton | 1/3 | 87.04 us | `compute_block_sums_kernel` |
| fp16 | triton | 2/3 | 5.44 us | `compute_mean_invstd_kernel` |
| fp16 | triton | 3/3 | 21.18 us | `apply_batch_norm_kernel` |
| fp32 | cutile | 1/3 | 86.11 us | `compute_block_sums_kernel_Kt1_A1f32_1i16t1_p16_A1f` |
| fp32 | cutile | 2/3 | 5.44 us | `compute_mean_invstd_kernel_Kt1_A1f32_1i16t1_p16_A1` |
| fp32 | cutile | 3/3 | 49.70 us | `apply_batch_norm_kernel_Kt1_A1f32_1i16t1_p16_A1f32` |
| fp32 | triton | 1/3 | 86.02 us | `compute_block_sums_kernel` |
| fp32 | triton | 2/3 | 5.22 us | `compute_mean_invstd_kernel` |
| fp32 | triton | 3/3 | 26.18 us | `apply_batch_norm_kernel` |

## Key findings (auto-derived)

- **fp16**: Triton is **1.23× faster** (113.7 µs vs 140.2 µs).
- **bf16**: Triton is **1.23× faster** (113.4 µs vs 139.1 µs).
- **fp32**: Triton is **1.20× faster** (117.4 µs vs 141.2 µs).

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
