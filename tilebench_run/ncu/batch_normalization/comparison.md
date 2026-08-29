# NCU Comparison: batch_normalization

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'C': 1024, 'eps': 1e-05, 'N': 20000}` | `{'ROWS': 8, 'num_warps': 8}` | `{'rows': 16, 'occupancy': 8}` |
| bf16 | `{'C': 1024, 'eps': 1e-05, 'N': 20000}` | `{'ROWS': 8, 'num_warps': 8}` | `{'rows': 16, 'occupancy': 8}` |
| fp32 | `{'C': 1024, 'eps': 1e-05, 'N': 20000}` | `{'ROWS': 8, 'num_warps': 8}` | `{'rows': 8, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 35.20 us | 40.28 % | 38.48 % | 56.34 % | 32.05 % | 21.24 % | 2.94 Tbyte/s | 256 | 48 register/thread | 0 byte/block | 4.10 Kbyte/block | 5 block / 12 block |
| fp16 | cutile | 51.07 us | 22.97 % | 22.97 % | 32.82 % | 19.80 % | 23.27 % | 1.76 Tbyte/s | 128 | 62 register/thread | 16.40 Kbyte/block | 0 byte/block | 8 block / 9 block |
| bf16 | triton | 34.85 us | 38.59 % | 38.59 % | 56.22 % | 32.54 % | 23.11 % | 2.95 Tbyte/s | 256 | 48 register/thread | 0 byte/block | 4.10 Kbyte/block | 5 block / 12 block |
| bf16 | cutile | 50.63 us | 22.68 % | 22.68 % | 32.91 % | 19.64 % | 23.07 % | 1.74 Tbyte/s | 128 | 62 register/thread | 16.40 Kbyte/block | 0 byte/block | 8 block / 9 block |
| fp32 | triton | 66.21 us | 43.09 % | 43.09 % | 7.19 % | 19.16 % | 6.73 % | 3.30 Tbyte/s | 128 | 132 register/thread | 0 byte/block | 0 byte/block | 3 block / 32 block |
| fp32 | cutile | 58.04 us | 64.83 % | 64.83 % | 47.56 % | 34.76 % | 22.01 % | 4.96 Tbyte/s | 128 | 92 register/thread | 0 byte/block | 0 byte/block | 5 block / 32 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| bf16 | cutile | 1/3 | 19.81 us | `compute_block_sums_kernel_Kt1_A2bf16_1v8l0_2t1_3i1` |
| bf16 | cutile | 2/3 | 7.52 us | `compute_mean_invstd_kernel_Kt1_A1f32_1i16t1_p16_A1` |
| bf16 | cutile | 3/3 | 23.30 us | `apply_batch_norm_kernel_Kt1_A2bf16_1v8l0_2t1_3i16_` |
| bf16 | triton | 1/3 | 13.76 us | `compute_block_sums_kernel` |
| bf16 | triton | 2/3 | 7.30 us | `compute_mean_invstd_kernel` |
| bf16 | triton | 3/3 | 13.79 us | `apply_batch_norm_kernel` |
| fp16 | cutile | 1/3 | 20.48 us | `compute_block_sums_kernel_Kt1_A2f16_1v8l0_2t1_3i16` |
| fp16 | cutile | 2/3 | 7.55 us | `compute_mean_invstd_kernel_Kt1_A1f32_1i16t1_p16_A1` |
| fp16 | cutile | 3/3 | 23.04 us | `apply_batch_norm_kernel_Kt1_A2f16_1v8l0_2t1_3i16_p` |
| fp16 | triton | 1/3 | 13.73 us | `compute_block_sums_kernel` |
| fp16 | triton | 2/3 | 7.49 us | `compute_mean_invstd_kernel` |
| fp16 | triton | 3/3 | 13.98 us | `apply_batch_norm_kernel` |
| fp32 | cutile | 1/3 | 24.38 us | `compute_block_sums_kernel_Kt1_A2f32_1v4l0_2t1_3i16` |
| fp32 | cutile | 2/3 | 7.74 us | `compute_mean_invstd_kernel_Kt1_A1f32_1i16t1_p16_A1` |
| fp32 | cutile | 3/3 | 25.92 us | `apply_batch_norm_kernel_Kt1_A2f32_1v4l0_2t1_3i16_p` |
| fp32 | triton | 1/3 | 37.18 us | `compute_block_sums_kernel` |
| fp32 | triton | 2/3 | 7.65 us | `compute_mean_invstd_kernel` |
| fp32 | triton | 3/3 | 21.38 us | `apply_batch_norm_kernel` |

## Key findings (auto-derived)

- **fp16**: Triton is **1.45× faster** (35.2 µs vs 51.1 µs).
- **bf16**: Triton is **1.45× faster** (34.9 µs vs 50.6 µs).
- **fp32**: cuTile is **1.14× faster** (58.0 µs vs 66.2 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **bf16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — Memory is more heavily utilized than Compute
- **fp32 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.70 full waves across all SMs. Look at Launch Statistics for more details.

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
