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
| fp16 | triton | 113.60 us | 91.23 % | 6.13 % | 97.49 % | 64.11 % | 12.59 % | 470.12 Gbyte/s | 128 | 26 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp16 | cutile | 140.10 us | 92.13 % | 6.16 % | 97.23 % | 64.46 % | 14.52 % | 472.60 Gbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| bf16 | triton | 113.31 us | 92.52 % | 6.14 % | 97.44 % | 64.25 % | 12.77 % | 470.63 Gbyte/s | 128 | 26 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| bf16 | cutile | 139.59 us | 91.47 % | 6.15 % | 97.31 % | 64.29 % | 14.41 % | 471.26 Gbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 117.24 us | 92.62 % | 12.88 % | 98.42 % | 67.31 % | 11.95 % | 987.89 Gbyte/s | 128 | 25 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp32 | cutile | 141.50 us | 92.09 % | 12.89 % | 98.24 % | 67.42 % | 13.61 % | 988.70 Gbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| bf16 | cutile | 1/3 | 86.98 us | `_compute_block_sums_kernel_Kt1_A1bf16_1i16t1_p16_A` |
| bf16 | cutile | 2/3 | 5.73 us | `_compute_mean_invstd_kernel_Kt1_A1f32_1i16t1_p16_A` |
| bf16 | cutile | 3/3 | 46.88 us | `_apply_batch_norm_kernel_Kt1_A1bf16_1i16t1_p16_A1b` |
| bf16 | triton | 1/3 | 87.23 us | `_compute_block_sums_kernel` |
| bf16 | triton | 2/3 | 5.18 us | `_compute_mean_invstd_kernel` |
| bf16 | triton | 3/3 | 20.90 us | `_apply_batch_norm_kernel` |
| fp16 | cutile | 1/3 | 86.88 us | `_compute_block_sums_kernel_Kt1_A1f16_1i16t1_p16_A1` |
| fp16 | cutile | 2/3 | 5.57 us | `_compute_mean_invstd_kernel_Kt1_A1f32_1i16t1_p16_A` |
| fp16 | cutile | 3/3 | 47.65 us | `_apply_batch_norm_kernel_Kt1_A1f16_1i16t1_p16_A1f1` |
| fp16 | triton | 1/3 | 87.30 us | `_compute_block_sums_kernel` |
| fp16 | triton | 2/3 | 5.15 us | `_compute_mean_invstd_kernel` |
| fp16 | triton | 3/3 | 21.15 us | `_apply_batch_norm_kernel` |
| fp32 | cutile | 1/3 | 86.43 us | `_compute_block_sums_kernel_Kt1_A1f32_1i16t1_p16_A1` |
| fp32 | cutile | 2/3 | 5.50 us | `_compute_mean_invstd_kernel_Kt1_A1f32_1i16t1_p16_A` |
| fp32 | cutile | 3/3 | 49.57 us | `_apply_batch_norm_kernel_Kt1_A1f32_1i16t1_p16_A1f3` |
| fp32 | triton | 1/3 | 86.56 us | `_compute_block_sums_kernel` |
| fp32 | triton | 2/3 | 5.34 us | `_compute_mean_invstd_kernel` |
| fp32 | triton | 3/3 | 25.34 us | `_apply_batch_norm_kernel` |

## Key findings (auto-derived)

- **fp16**: Triton is **1.23× faster** (113.6 µs vs 140.1 µs).
- **bf16**: Triton is **1.23× faster** (113.3 µs vs 139.6 µs).
- **fp32**: Triton is **1.21× faster** (117.2 µs vs 141.5 µs).

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
