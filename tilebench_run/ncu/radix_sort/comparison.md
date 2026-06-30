# NCU Comparison: radix_sort

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| int32 | `{'n': 20000000}` | `{'num_warps': 4}` | `{'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| int32 | triton | 2390.19 us | 70.19 % | 31.98 % | 77.57 % | 21.07 % | 53.98 % | 2.45 Tbyte/s | 128 | 48 register/thread | 0 byte/block | 8.19 Kbyte/block | 10 block / 14 block |
| int32 | cutile | 187.78 us | 73.39 % | 23.63 % | 78.76 % | 19.71 % | 71.67 % | 1.81 Tbyte/s | 128 | 64 register/thread | 4.11 Kbyte/block | 0 byte/block | 8 block / 19 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| int32 | cutile | 1/10 | 16.80 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 2/10 | 4.74 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 3/10 | 4.74 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 4/10 | 4.83 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 5/10 | 63.07 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 6/10 | 16.70 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 7/10 | 4.51 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 8/10 | 4.35 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 9/10 | 4.90 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 10/10 | 63.14 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | triton | 1/160 | 4.93 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 2/160 | 44.64 us | `_radix_sort_kernel` |
| int32 | triton | 3/160 | 15.58 us | `_count_ones_in_block` |
| int32 | triton | 4/160 | 4.54 us | `_count_ones_per_block_blocks` |
| int32 | triton | 5/160 | 4.48 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 6/160 | 4.70 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 7/160 | 45.22 us | `_radix_sort_kernel` |
| int32 | triton | 8/160 | 15.84 us | `_count_ones_in_block` |
| int32 | triton | 9/160 | 4.51 us | `_count_ones_per_block_blocks` |
| int32 | triton | 10/160 | 4.10 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 11/160 | 4.61 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 12/160 | 46.40 us | `_radix_sort_kernel` |
| int32 | triton | 13/160 | 15.87 us | `_count_ones_in_block` |
| int32 | triton | 14/160 | 4.74 us | `_count_ones_per_block_blocks` |
| int32 | triton | 15/160 | 4.32 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 16/160 | 4.96 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 17/160 | 46.05 us | `_radix_sort_kernel` |
| int32 | triton | 18/160 | 15.74 us | `_count_ones_in_block` |
| int32 | triton | 19/160 | 4.45 us | `_count_ones_per_block_blocks` |
| int32 | triton | 20/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 21/160 | 4.80 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 22/160 | 45.25 us | `_radix_sort_kernel` |
| int32 | triton | 23/160 | 15.68 us | `_count_ones_in_block` |
| int32 | triton | 24/160 | 4.29 us | `_count_ones_per_block_blocks` |
| int32 | triton | 25/160 | 4.22 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 26/160 | 4.51 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 27/160 | 45.34 us | `_radix_sort_kernel` |
| int32 | triton | 28/160 | 15.71 us | `_count_ones_in_block` |
| int32 | triton | 29/160 | 4.67 us | `_count_ones_per_block_blocks` |
| int32 | triton | 30/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 31/160 | 4.64 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 32/160 | 45.47 us | `_radix_sort_kernel` |
| int32 | triton | 33/160 | 15.74 us | `_count_ones_in_block` |
| int32 | triton | 34/160 | 4.61 us | `_count_ones_per_block_blocks` |
| int32 | triton | 35/160 | 4.42 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 36/160 | 4.83 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 37/160 | 45.34 us | `_radix_sort_kernel` |
| int32 | triton | 38/160 | 15.78 us | `_count_ones_in_block` |
| int32 | triton | 39/160 | 4.42 us | `_count_ones_per_block_blocks` |
| int32 | triton | 40/160 | 4.19 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 41/160 | 4.90 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 42/160 | 45.28 us | `_radix_sort_kernel` |
| int32 | triton | 43/160 | 15.65 us | `_count_ones_in_block` |
| int32 | triton | 44/160 | 4.48 us | `_count_ones_per_block_blocks` |
| int32 | triton | 45/160 | 4.26 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 46/160 | 4.48 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 47/160 | 45.15 us | `_radix_sort_kernel` |
| int32 | triton | 48/160 | 15.78 us | `_count_ones_in_block` |
| int32 | triton | 49/160 | 4.42 us | `_count_ones_per_block_blocks` |
| int32 | triton | 50/160 | 4.22 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 51/160 | 4.83 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 52/160 | 45.50 us | `_radix_sort_kernel` |
| int32 | triton | 53/160 | 15.74 us | `_count_ones_in_block` |
| int32 | triton | 54/160 | 4.77 us | `_count_ones_per_block_blocks` |
| int32 | triton | 55/160 | 4.35 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 56/160 | 4.83 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 57/160 | 45.38 us | `_radix_sort_kernel` |
| int32 | triton | 58/160 | 15.71 us | `_count_ones_in_block` |
| int32 | triton | 59/160 | 4.70 us | `_count_ones_per_block_blocks` |
| int32 | triton | 60/160 | 4.29 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 61/160 | 4.64 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 62/160 | 45.57 us | `_radix_sort_kernel` |
| int32 | triton | 63/160 | 15.55 us | `_count_ones_in_block` |
| int32 | triton | 64/160 | 4.42 us | `_count_ones_per_block_blocks` |
| int32 | triton | 65/160 | 4.06 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 66/160 | 4.45 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 67/160 | 45.22 us | `_radix_sort_kernel` |
| int32 | triton | 68/160 | 15.78 us | `_count_ones_in_block` |
| int32 | triton | 69/160 | 4.51 us | `_count_ones_per_block_blocks` |
| int32 | triton | 70/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 71/160 | 4.64 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 72/160 | 45.31 us | `_radix_sort_kernel` |
| int32 | triton | 73/160 | 15.84 us | `_count_ones_in_block` |
| int32 | triton | 74/160 | 4.67 us | `_count_ones_per_block_blocks` |
| int32 | triton | 75/160 | 4.26 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 76/160 | 4.86 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 77/160 | 45.63 us | `_radix_sort_kernel` |
| int32 | triton | 78/160 | 15.81 us | `_count_ones_in_block` |
| int32 | triton | 79/160 | 4.67 us | `_count_ones_per_block_blocks` |
| int32 | triton | 80/160 | 4.22 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 81/160 | 4.67 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 82/160 | 45.50 us | `_radix_sort_kernel` |
| int32 | triton | 83/160 | 15.46 us | `_count_ones_in_block` |
| int32 | triton | 84/160 | 4.32 us | `_count_ones_per_block_blocks` |
| int32 | triton | 85/160 | 4.03 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 86/160 | 4.80 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 87/160 | 45.73 us | `_radix_sort_kernel` |
| int32 | triton | 88/160 | 15.58 us | `_count_ones_in_block` |
| int32 | triton | 89/160 | 4.45 us | `_count_ones_per_block_blocks` |
| int32 | triton | 90/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 91/160 | 4.64 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 92/160 | 45.47 us | `_radix_sort_kernel` |
| int32 | triton | 93/160 | 15.81 us | `_count_ones_in_block` |
| int32 | triton | 94/160 | 4.58 us | `_count_ones_per_block_blocks` |
| int32 | triton | 95/160 | 4.45 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 96/160 | 4.80 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 97/160 | 45.82 us | `_radix_sort_kernel` |
| int32 | triton | 98/160 | 15.58 us | `_count_ones_in_block` |
| int32 | triton | 99/160 | 4.70 us | `_count_ones_per_block_blocks` |
| int32 | triton | 100/160 | 4.19 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 101/160 | 4.86 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 102/160 | 45.41 us | `_radix_sort_kernel` |
| int32 | triton | 103/160 | 15.65 us | `_count_ones_in_block` |
| int32 | triton | 104/160 | 4.26 us | `_count_ones_per_block_blocks` |
| int32 | triton | 105/160 | 4.26 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 106/160 | 4.54 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 107/160 | 45.18 us | `_radix_sort_kernel` |
| int32 | triton | 108/160 | 16.03 us | `_count_ones_in_block` |
| int32 | triton | 109/160 | 4.38 us | `_count_ones_per_block_blocks` |
| int32 | triton | 110/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 111/160 | 4.61 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 112/160 | 45.28 us | `_radix_sort_kernel` |
| int32 | triton | 113/160 | 15.84 us | `_count_ones_in_block` |
| int32 | triton | 114/160 | 4.61 us | `_count_ones_per_block_blocks` |
| int32 | triton | 115/160 | 4.26 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 116/160 | 4.90 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 117/160 | 45.38 us | `_radix_sort_kernel` |
| int32 | triton | 118/160 | 15.62 us | `_count_ones_in_block` |
| int32 | triton | 119/160 | 4.48 us | `_count_ones_per_block_blocks` |
| int32 | triton | 120/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 121/160 | 4.67 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 122/160 | 45.57 us | `_radix_sort_kernel` |
| int32 | triton | 123/160 | 15.49 us | `_count_ones_in_block` |
| int32 | triton | 124/160 | 4.42 us | `_count_ones_per_block_blocks` |
| int32 | triton | 125/160 | 4.64 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 126/160 | 4.67 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 127/160 | 45.28 us | `_radix_sort_kernel` |
| int32 | triton | 128/160 | 15.71 us | `_count_ones_in_block` |
| int32 | triton | 129/160 | 4.51 us | `_count_ones_per_block_blocks` |
| int32 | triton | 130/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 131/160 | 4.77 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 132/160 | 45.18 us | `_radix_sort_kernel` |
| int32 | triton | 133/160 | 16.48 us | `_count_ones_in_block` |
| int32 | triton | 134/160 | 4.61 us | `_count_ones_per_block_blocks` |
| int32 | triton | 135/160 | 4.42 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 136/160 | 4.99 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 137/160 | 45.34 us | `_radix_sort_kernel` |
| int32 | triton | 138/160 | 15.62 us | `_count_ones_in_block` |
| int32 | triton | 139/160 | 4.51 us | `_count_ones_per_block_blocks` |
| int32 | triton | 140/160 | 4.22 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 141/160 | 5.09 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 142/160 | 45.50 us | `_radix_sort_kernel` |
| int32 | triton | 143/160 | 15.68 us | `_count_ones_in_block` |
| int32 | triton | 144/160 | 4.51 us | `_count_ones_per_block_blocks` |
| int32 | triton | 145/160 | 4.03 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 146/160 | 4.54 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 147/160 | 45.41 us | `_radix_sort_kernel` |
| int32 | triton | 148/160 | 15.74 us | `_count_ones_in_block` |
| int32 | triton | 149/160 | 4.48 us | `_count_ones_per_block_blocks` |
| int32 | triton | 150/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 151/160 | 4.61 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 152/160 | 45.31 us | `_radix_sort_kernel` |
| int32 | triton | 153/160 | 16.06 us | `_count_ones_in_block` |
| int32 | triton | 154/160 | 4.70 us | `_count_ones_per_block_blocks` |
| int32 | triton | 155/160 | 4.45 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 156/160 | 4.90 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 157/160 | 45.38 us | `_radix_sort_kernel` |
| int32 | triton | 158/160 | 15.58 us | `_count_ones_in_block` |
| int32 | triton | 159/160 | 4.64 us | `_count_ones_per_block_blocks` |
| int32 | triton | 160/160 | 4.80 us | `_compute_prefix_sums_per_block_of_blocks` |

## Key findings (auto-derived)

- **int32**: cuTile is **12.73× faster** (187.8 µs vs 2390.2 µs).

## NCU's own bottleneck verdict

- **int32 / cutile** — Compute and Memory are well-balanced
- **int32 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_int32.ncu-rep`
- `triton_int32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
