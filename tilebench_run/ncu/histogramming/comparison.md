# NCU Comparison: histogramming

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| int32 | `{'N': 67108864, 'num_bins': 4096}` | `{'partial_BLOCK_SIZE': 2048, 'partial_num_warps': 8, 'reduce_BLOCK_ROWS': 128, 'reduce_BLOCK_BINS': 64, 'reduce_num_warps': 8, 'reduce_num_stages': 2}` | `{'partial_block_size': 2048, 'partial_occupancy': 4, 'reduce_block_rows': 128, 'reduce_block_bins': 64, 'reduce_occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| int32 | triton | 1853.42 us | 5.39 % | 1.97 % | 6.87 % | 2.18 % | 2.40 % | 150.79 Gbyte/s | 128 | 40 register/thread | 0 byte/block | 8.19 Kbyte/block | 12 block / 14 block |
| int32 | cutile | 1972.42 us | 5.50 % | 1.81 % | 6.81 % | 2.00 % | 4.59 % | 138.96 Gbyte/s | 128 | 53 register/thread | 13.32 Kbyte/block | 0 byte/block | 9 block / 9 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| int32 | cutile | 1/3 | 1960.00 us | `_histogram_partial_kernel_Kt1_A1i32_1i16t1_p16_A2i` |
| int32 | cutile | 2/3 | 8.10 us | `_histogram_reduce_kernel_Kt1_A2i32_1v4l0_2t1_3i16_` |
| int32 | cutile | 3/3 | 4.32 us | `void at::vectorized_elementwise_kernel<4, at::Fill` |
| int32 | triton | 1/3 | 1810.00 us | `_histogram_partial_kernel` |
| int32 | triton | 2/3 | 39.20 us | `_histogram_reduce_kernel` |
| int32 | triton | 3/3 | 4.22 us | `void at::vectorized_elementwise_kernel<4, at::Fill` |

## Key findings (auto-derived)

- **int32**: Triton is **1.06× faster** (1853.4 µs vs 1972.4 µs).

## NCU's own bottleneck verdict

- **int32 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.19 full waves across all SMs. Look at Launch Statistics for more details.
- **int32 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.14 full waves across all SMs. Look at Launch Statistics for more details.

## Reports

- `cutile_int32.ncu-rep`
- `triton_int32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
