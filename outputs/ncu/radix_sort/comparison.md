# NCU Comparison: radix_sort

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| int32 | `{'n': 20000000}` | `{'num_warps': 4}` | `{'occupancy': 32}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| int32 | triton | 1331.18 us | 72.78 % | 28.95 % | 79.91 % | 20.58 % | 62.99 % | 2.22 Tbyte/s | 128 | 64 register/thread | 0 byte/block | 8.19 Kbyte/block | 8 block / 14 block |
| int32 | cutile | 2529.92 us | 72.02 % | 12.49 % | 74.65 % | 10.10 % | 58.27 % | 958.21 Gbyte/s | 128 | 108 register/thread | 33.80 Kbyte/block | 0 byte/block | 4 block / 4 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| int32 | cutile | 1/80 | 26.91 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 2/80 | 4.90 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 3/80 | 4.48 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 4/80 | 5.22 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 5/80 | 117.79 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | cutile | 6/80 | 25.41 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 7/80 | 4.83 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 8/80 | 4.51 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 9/80 | 5.22 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 10/80 | 117.34 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | cutile | 11/80 | 26.59 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 12/80 | 4.93 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 13/80 | 4.51 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 14/80 | 5.22 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 15/80 | 118.11 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | cutile | 16/80 | 25.89 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 17/80 | 4.70 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 18/80 | 4.61 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 19/80 | 5.22 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 20/80 | 117.41 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | cutile | 21/80 | 26.50 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 22/80 | 4.70 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 23/80 | 5.31 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 24/80 | 5.18 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 25/80 | 117.86 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | cutile | 26/80 | 25.63 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 27/80 | 4.80 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 28/80 | 4.51 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 29/80 | 4.96 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 30/80 | 117.34 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | cutile | 31/80 | 26.56 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 32/80 | 4.93 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 33/80 | 4.51 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 34/80 | 5.02 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 35/80 | 117.89 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | cutile | 36/80 | 25.38 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 37/80 | 4.70 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 38/80 | 4.48 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 39/80 | 5.18 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 40/80 | 117.22 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | cutile | 41/80 | 26.82 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 42/80 | 4.80 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 43/80 | 4.51 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 44/80 | 5.06 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 45/80 | 118.05 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | cutile | 46/80 | 25.92 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 47/80 | 5.15 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 48/80 | 4.74 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 49/80 | 5.18 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 50/80 | 117.31 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | cutile | 51/80 | 26.69 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 52/80 | 4.74 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 53/80 | 4.51 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 54/80 | 5.06 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 55/80 | 117.79 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | cutile | 56/80 | 25.44 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 57/80 | 4.74 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 58/80 | 4.61 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 59/80 | 5.15 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 60/80 | 116.90 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | cutile | 61/80 | 27.04 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 62/80 | 4.80 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 63/80 | 4.48 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 64/80 | 5.18 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 65/80 | 117.57 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | cutile | 66/80 | 25.47 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 67/80 | 4.77 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 68/80 | 4.48 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 69/80 | 5.60 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 70/80 | 117.25 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | cutile | 71/80 | 26.40 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 72/80 | 4.90 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 73/80 | 4.80 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 74/80 | 5.06 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 75/80 | 117.44 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | cutile | 76/80 | 25.54 us | `radix_histogram_kernel_Kt1_A1i32_1i16t1_p16_A1i32_` |
| int32 | cutile | 77/80 | 4.93 us | `radix_sum_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i32` |
| int32 | cutile | 78/80 | 4.64 us | `radix_scan_chunk_sums_kernel_Kt1_A1i32_1t1_p16_I10` |
| int32 | cutile | 79/80 | 5.06 us | `radix_scan_chunks_kernel_Kt1_A1i32_1i16t1_p16_A1i3` |
| int32 | cutile | 80/80 | 114.88 us | `radix_scatter_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i` |
| int32 | triton | 1/80 | 18.98 us | `radix_histogram_kernel` |
| int32 | triton | 2/80 | 4.32 us | `radix_sum_chunks_kernel` |
| int32 | triton | 3/80 | 4.26 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 4/80 | 4.64 us | `radix_scan_chunks_kernel` |
| int32 | triton | 5/80 | 51.33 us | `radix_scatter_kernel` |
| int32 | triton | 6/80 | 18.66 us | `radix_histogram_kernel` |
| int32 | triton | 7/80 | 4.58 us | `radix_sum_chunks_kernel` |
| int32 | triton | 8/80 | 4.22 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 9/80 | 4.42 us | `radix_scan_chunks_kernel` |
| int32 | triton | 10/80 | 51.17 us | `radix_scatter_kernel` |
| int32 | triton | 11/80 | 19.07 us | `radix_histogram_kernel` |
| int32 | triton | 12/80 | 4.51 us | `radix_sum_chunks_kernel` |
| int32 | triton | 13/80 | 4.29 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 14/80 | 4.58 us | `radix_scan_chunks_kernel` |
| int32 | triton | 15/80 | 51.10 us | `radix_scatter_kernel` |
| int32 | triton | 16/80 | 18.66 us | `radix_histogram_kernel` |
| int32 | triton | 17/80 | 4.32 us | `radix_sum_chunks_kernel` |
| int32 | triton | 18/80 | 4.26 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 19/80 | 4.51 us | `radix_scan_chunks_kernel` |
| int32 | triton | 20/80 | 51.26 us | `radix_scatter_kernel` |
| int32 | triton | 21/80 | 19.71 us | `radix_histogram_kernel` |
| int32 | triton | 22/80 | 4.61 us | `radix_sum_chunks_kernel` |
| int32 | triton | 23/80 | 4.38 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 24/80 | 4.67 us | `radix_scan_chunks_kernel` |
| int32 | triton | 25/80 | 51.26 us | `radix_scatter_kernel` |
| int32 | triton | 26/80 | 18.69 us | `radix_histogram_kernel` |
| int32 | triton | 27/80 | 4.54 us | `radix_sum_chunks_kernel` |
| int32 | triton | 28/80 | 4.35 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 29/80 | 4.42 us | `radix_scan_chunks_kernel` |
| int32 | triton | 30/80 | 51.14 us | `radix_scatter_kernel` |
| int32 | triton | 31/80 | 18.94 us | `radix_histogram_kernel` |
| int32 | triton | 32/80 | 4.32 us | `radix_sum_chunks_kernel` |
| int32 | triton | 33/80 | 4.26 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 34/80 | 4.42 us | `radix_scan_chunks_kernel` |
| int32 | triton | 35/80 | 51.07 us | `radix_scatter_kernel` |
| int32 | triton | 36/80 | 18.69 us | `radix_histogram_kernel` |
| int32 | triton | 37/80 | 4.32 us | `radix_sum_chunks_kernel` |
| int32 | triton | 38/80 | 4.22 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 39/80 | 4.38 us | `radix_scan_chunks_kernel` |
| int32 | triton | 40/80 | 50.85 us | `radix_scatter_kernel` |
| int32 | triton | 41/80 | 18.94 us | `radix_histogram_kernel` |
| int32 | triton | 42/80 | 4.35 us | `radix_sum_chunks_kernel` |
| int32 | triton | 43/80 | 4.26 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 44/80 | 4.38 us | `radix_scan_chunks_kernel` |
| int32 | triton | 45/80 | 51.20 us | `radix_scatter_kernel` |
| int32 | triton | 46/80 | 18.88 us | `radix_histogram_kernel` |
| int32 | triton | 47/80 | 4.93 us | `radix_sum_chunks_kernel` |
| int32 | triton | 48/80 | 4.26 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 49/80 | 4.38 us | `radix_scan_chunks_kernel` |
| int32 | triton | 50/80 | 51.10 us | `radix_scatter_kernel` |
| int32 | triton | 51/80 | 18.94 us | `radix_histogram_kernel` |
| int32 | triton | 52/80 | 4.58 us | `radix_sum_chunks_kernel` |
| int32 | triton | 53/80 | 4.26 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 54/80 | 4.64 us | `radix_scan_chunks_kernel` |
| int32 | triton | 55/80 | 51.07 us | `radix_scatter_kernel` |
| int32 | triton | 56/80 | 18.78 us | `radix_histogram_kernel` |
| int32 | triton | 57/80 | 4.54 us | `radix_sum_chunks_kernel` |
| int32 | triton | 58/80 | 4.29 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 59/80 | 4.38 us | `radix_scan_chunks_kernel` |
| int32 | triton | 60/80 | 51.30 us | `radix_scatter_kernel` |
| int32 | triton | 61/80 | 18.82 us | `radix_histogram_kernel` |
| int32 | triton | 62/80 | 4.58 us | `radix_sum_chunks_kernel` |
| int32 | triton | 63/80 | 4.26 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 64/80 | 4.61 us | `radix_scan_chunks_kernel` |
| int32 | triton | 65/80 | 51.20 us | `radix_scatter_kernel` |
| int32 | triton | 66/80 | 18.56 us | `radix_histogram_kernel` |
| int32 | triton | 67/80 | 4.58 us | `radix_sum_chunks_kernel` |
| int32 | triton | 68/80 | 4.45 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 69/80 | 4.38 us | `radix_scan_chunks_kernel` |
| int32 | triton | 70/80 | 51.07 us | `radix_scatter_kernel` |
| int32 | triton | 71/80 | 19.04 us | `radix_histogram_kernel` |
| int32 | triton | 72/80 | 4.32 us | `radix_sum_chunks_kernel` |
| int32 | triton | 73/80 | 4.35 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 74/80 | 4.64 us | `radix_scan_chunks_kernel` |
| int32 | triton | 75/80 | 51.07 us | `radix_scatter_kernel` |
| int32 | triton | 76/80 | 18.62 us | `radix_histogram_kernel` |
| int32 | triton | 77/80 | 4.58 us | `radix_sum_chunks_kernel` |
| int32 | triton | 78/80 | 4.35 us | `radix_scan_chunk_sums_kernel` |
| int32 | triton | 79/80 | 4.42 us | `radix_scan_chunks_kernel` |
| int32 | triton | 80/80 | 49.44 us | `radix_scatter_kernel` |

## Key findings (auto-derived)

- **int32**: Triton is **1.90× faster** (1331.2 µs vs 2529.9 µs).

## NCU's own bottleneck verdict

- **int32 / cutile** — Memory is more heavily utilized than Compute
- **int32 / triton** — Compute and Memory are well-balanced

## Reports

- `cutile_int32.ncu-rep`
- `triton_int32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
