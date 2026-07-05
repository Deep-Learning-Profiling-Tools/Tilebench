# NCU Comparison: histogramming

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| int32 | `{'N': 67108864, 'num_bins': 4096}` | `{'partial_BLOCK_SIZE': 2048, 'partial_num_warps': 8, 'reduce_BLOCK_ROWS': 128, 'reduce_BLOCK_BINS': 64, 'reduce_num_warps': 8, 'reduce_num_stages': 2}` | `{'partial_block_size': 2048, 'partial_occupancy': 4, 'reduce_block_rows': 128, 'reduce_block_bins': 64, 'reduce_occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| int32 | triton | 799.13 us | 12.16 % | 4.51 % | 13.49 % | 4.78 % | 5.48 % | 346.27 Gbyte/s | 256 | 40 register/thread | 0 byte/block | 16.38 Kbyte/block | 6 block / 7 block |
| int32 | cutile | 1436.78 us | 7.44 % | 2.50 % | 7.84 % | 2.75 % | 4.93 % | 191.73 Gbyte/s | 128 | 128 register/thread | 26.64 Kbyte/block | 0 byte/block | 4 block / 4 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| int32 | cutile | 1/2 | 1430.00 us | `histogram_partial_kernel_Kt1_A1i32_1i16t1_p16_A2i3` |
| int32 | cutile | 2/2 | 6.78 us | `histogram_reduce_kernel_Kt1_A2i32_1v4l0_2t1_3i16_p` |
| int32 | triton | 1/2 | 790.14 us | `histogram_partial_kernel` |
| int32 | triton | 2/2 | 8.99 us | `histogram_reduce_kernel` |

## Key findings (auto-derived)

- **int32**: Triton is **1.80× faster** (799.1 µs vs 1436.8 µs).

## NCU's own bottleneck verdict

- **int32 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.43 full waves across all SMs. Look at Launch Statistics for more details.
- **int32 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.29 full waves across all SMs. Look at Launch Statistics for more details.

## Reports

- `cutile_int32.ncu-rep`
- `triton_int32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
