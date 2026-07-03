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
| int32 | triton | 2371.37 us | 70.78 % | 32.01 % | 77.61 % | 21.10 % | 54.41 % | 2.45 Tbyte/s | 128 | 48 register/thread | 0 byte/block | 8.19 Kbyte/block | 10 block / 14 block |
| int32 | cutile | 2978.68 us | 72.67 % | 23.52 % | 78.40 % | 19.64 % | 70.98 % | 1.80 Tbyte/s | 128 | 64 register/thread | 4.11 Kbyte/block | 0 byte/block | 8 block / 19 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| int32 | cutile | 1/160 | 16.96 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 2/160 | 4.48 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 3/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 4/160 | 4.77 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 5/160 | 63.01 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 6/160 | 16.54 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 7/160 | 4.51 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 8/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 9/160 | 5.57 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 10/160 | 62.94 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 11/160 | 17.09 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 12/160 | 4.48 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 13/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 14/160 | 4.74 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 15/160 | 63.07 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 16/160 | 16.51 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 17/160 | 4.54 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 18/160 | 4.26 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 19/160 | 4.96 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 20/160 | 62.72 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 21/160 | 17.12 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 22/160 | 4.58 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 23/160 | 4.22 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 24/160 | 4.96 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 25/160 | 62.85 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 26/160 | 16.32 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 27/160 | 4.80 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 28/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 29/160 | 4.80 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 30/160 | 62.88 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 31/160 | 16.64 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 32/160 | 4.64 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 33/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 34/160 | 4.99 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 35/160 | 63.33 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 36/160 | 16.48 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 37/160 | 4.48 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 38/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 39/160 | 4.74 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 40/160 | 62.91 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 41/160 | 16.54 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 42/160 | 4.48 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 43/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 44/160 | 4.77 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 45/160 | 63.07 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 46/160 | 16.83 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 47/160 | 4.48 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 48/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 49/160 | 4.74 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 50/160 | 62.91 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 51/160 | 16.54 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 52/160 | 4.51 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 53/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 54/160 | 4.83 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 55/160 | 63.01 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 56/160 | 16.48 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 57/160 | 4.48 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 58/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 59/160 | 5.02 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 60/160 | 62.88 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 61/160 | 16.45 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 62/160 | 4.51 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 63/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 64/160 | 4.70 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 65/160 | 62.98 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 66/160 | 16.86 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 67/160 | 4.48 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 68/160 | 4.22 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 69/160 | 4.99 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 70/160 | 62.85 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 71/160 | 16.64 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 72/160 | 4.51 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 73/160 | 4.10 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 74/160 | 4.74 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 75/160 | 62.91 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 76/160 | 16.45 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 77/160 | 4.67 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 78/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 79/160 | 4.93 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 80/160 | 62.78 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 81/160 | 16.48 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 82/160 | 4.70 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 83/160 | 4.22 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 84/160 | 4.96 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 85/160 | 63.01 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 86/160 | 16.58 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 87/160 | 4.51 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 88/160 | 4.19 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 89/160 | 4.70 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 90/160 | 62.98 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 91/160 | 16.51 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 92/160 | 4.51 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 93/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 94/160 | 4.74 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 95/160 | 62.88 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 96/160 | 16.48 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 97/160 | 4.48 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 98/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 99/160 | 4.74 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 100/160 | 63.01 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 101/160 | 16.48 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 102/160 | 4.51 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 103/160 | 4.10 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 104/160 | 4.74 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 105/160 | 62.85 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 106/160 | 16.58 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 107/160 | 4.67 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 108/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 109/160 | 4.74 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 110/160 | 62.82 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 111/160 | 16.51 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 112/160 | 4.48 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 113/160 | 4.10 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 114/160 | 4.77 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 115/160 | 62.98 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 116/160 | 16.54 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 117/160 | 5.06 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 118/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 119/160 | 4.77 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 120/160 | 62.72 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 121/160 | 16.93 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 122/160 | 4.54 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 123/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 124/160 | 4.70 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 125/160 | 62.98 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 126/160 | 16.54 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 127/160 | 4.74 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 128/160 | 4.22 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 129/160 | 4.74 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 130/160 | 62.91 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 131/160 | 16.48 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 132/160 | 4.54 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 133/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 134/160 | 4.99 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 135/160 | 62.98 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 136/160 | 16.58 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 137/160 | 4.51 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 138/160 | 4.29 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 139/160 | 4.74 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 140/160 | 62.72 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 141/160 | 16.45 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 142/160 | 4.77 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 143/160 | 4.10 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 144/160 | 4.74 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 145/160 | 63.04 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 146/160 | 16.48 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 147/160 | 4.67 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 148/160 | 4.26 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 149/160 | 4.90 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 150/160 | 62.82 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 151/160 | 16.54 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 152/160 | 4.58 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 153/160 | 4.10 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 154/160 | 4.96 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 155/160 | 62.59 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | cutile | 156/160 | 16.74 us | `_count_ones_in_block_Kt1_A1i32_1i16t1_p16_A1i32_1t` |
| int32 | cutile | 157/160 | 4.54 us | `_count_ones_per_block_blocks_Kt1_A1i32_1t1_p16_A1i` |
| int32 | cutile | 158/160 | 4.26 us | `_compute_prefix_sums_per_block_of_blocks_Kt1_A1i32` |
| int32 | cutile | 159/160 | 5.12 us | `_compute_prefix_sums_per_block_Kt1_A1i32_1t1_p16_A` |
| int32 | cutile | 160/160 | 61.92 us | `_radix_sort_kernel_Kt1_A1i32_1i16t1_p16_A1i32_1i16` |
| int32 | triton | 1/160 | 15.78 us | `_count_ones_in_block` |
| int32 | triton | 2/160 | 4.29 us | `_count_ones_per_block_blocks` |
| int32 | triton | 3/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 4/160 | 4.54 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 5/160 | 45.57 us | `_radix_sort_kernel` |
| int32 | triton | 6/160 | 15.58 us | `_count_ones_in_block` |
| int32 | triton | 7/160 | 4.26 us | `_count_ones_per_block_blocks` |
| int32 | triton | 8/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 9/160 | 4.51 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 10/160 | 46.37 us | `_radix_sort_kernel` |
| int32 | triton | 11/160 | 15.68 us | `_count_ones_in_block` |
| int32 | triton | 12/160 | 4.26 us | `_count_ones_per_block_blocks` |
| int32 | triton | 13/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 14/160 | 4.48 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 15/160 | 45.41 us | `_radix_sort_kernel` |
| int32 | triton | 16/160 | 15.46 us | `_count_ones_in_block` |
| int32 | triton | 17/160 | 4.32 us | `_count_ones_per_block_blocks` |
| int32 | triton | 18/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 19/160 | 4.42 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 20/160 | 45.41 us | `_radix_sort_kernel` |
| int32 | triton | 21/160 | 15.52 us | `_count_ones_in_block` |
| int32 | triton | 22/160 | 4.16 us | `_count_ones_per_block_blocks` |
| int32 | triton | 23/160 | 4.26 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 24/160 | 4.67 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 25/160 | 45.57 us | `_radix_sort_kernel` |
| int32 | triton | 26/160 | 15.71 us | `_count_ones_in_block` |
| int32 | triton | 27/160 | 4.29 us | `_count_ones_per_block_blocks` |
| int32 | triton | 28/160 | 4.19 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 29/160 | 4.51 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 30/160 | 45.41 us | `_radix_sort_kernel` |
| int32 | triton | 31/160 | 15.65 us | `_count_ones_in_block` |
| int32 | triton | 32/160 | 4.29 us | `_count_ones_per_block_blocks` |
| int32 | triton | 33/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 34/160 | 4.90 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 35/160 | 45.41 us | `_radix_sort_kernel` |
| int32 | triton | 36/160 | 15.46 us | `_count_ones_in_block` |
| int32 | triton | 37/160 | 4.35 us | `_count_ones_per_block_blocks` |
| int32 | triton | 38/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 39/160 | 4.45 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 40/160 | 45.50 us | `_radix_sort_kernel` |
| int32 | triton | 41/160 | 15.49 us | `_count_ones_in_block` |
| int32 | triton | 42/160 | 4.26 us | `_count_ones_per_block_blocks` |
| int32 | triton | 43/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 44/160 | 4.70 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 45/160 | 45.47 us | `_radix_sort_kernel` |
| int32 | triton | 46/160 | 15.49 us | `_count_ones_in_block` |
| int32 | triton | 47/160 | 4.26 us | `_count_ones_per_block_blocks` |
| int32 | triton | 48/160 | 4.29 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 49/160 | 4.45 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 50/160 | 45.38 us | `_radix_sort_kernel` |
| int32 | triton | 51/160 | 15.81 us | `_count_ones_in_block` |
| int32 | triton | 52/160 | 4.48 us | `_count_ones_per_block_blocks` |
| int32 | triton | 53/160 | 4.22 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 54/160 | 4.45 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 55/160 | 45.41 us | `_radix_sort_kernel` |
| int32 | triton | 56/160 | 15.49 us | `_count_ones_in_block` |
| int32 | triton | 57/160 | 4.26 us | `_count_ones_per_block_blocks` |
| int32 | triton | 58/160 | 4.35 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 59/160 | 4.45 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 60/160 | 45.76 us | `_radix_sort_kernel` |
| int32 | triton | 61/160 | 15.55 us | `_count_ones_in_block` |
| int32 | triton | 62/160 | 4.22 us | `_count_ones_per_block_blocks` |
| int32 | triton | 63/160 | 4.19 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 64/160 | 4.48 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 65/160 | 45.57 us | `_radix_sort_kernel` |
| int32 | triton | 66/160 | 15.46 us | `_count_ones_in_block` |
| int32 | triton | 67/160 | 4.29 us | `_count_ones_per_block_blocks` |
| int32 | triton | 68/160 | 3.94 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 69/160 | 4.45 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 70/160 | 45.18 us | `_radix_sort_kernel` |
| int32 | triton | 71/160 | 16.26 us | `_count_ones_in_block` |
| int32 | triton | 72/160 | 4.29 us | `_count_ones_per_block_blocks` |
| int32 | triton | 73/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 74/160 | 4.45 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 75/160 | 45.47 us | `_radix_sort_kernel` |
| int32 | triton | 76/160 | 16.22 us | `_count_ones_in_block` |
| int32 | triton | 77/160 | 4.70 us | `_count_ones_per_block_blocks` |
| int32 | triton | 78/160 | 4.19 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 79/160 | 4.45 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 80/160 | 45.22 us | `_radix_sort_kernel` |
| int32 | triton | 81/160 | 15.58 us | `_count_ones_in_block` |
| int32 | triton | 82/160 | 4.64 us | `_count_ones_per_block_blocks` |
| int32 | triton | 83/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 84/160 | 4.70 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 85/160 | 45.41 us | `_radix_sort_kernel` |
| int32 | triton | 86/160 | 15.81 us | `_count_ones_in_block` |
| int32 | triton | 87/160 | 4.42 us | `_count_ones_per_block_blocks` |
| int32 | triton | 88/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 89/160 | 4.83 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 90/160 | 45.28 us | `_radix_sort_kernel` |
| int32 | triton | 91/160 | 15.62 us | `_count_ones_in_block` |
| int32 | triton | 92/160 | 4.26 us | `_count_ones_per_block_blocks` |
| int32 | triton | 93/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 94/160 | 4.42 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 95/160 | 45.28 us | `_radix_sort_kernel` |
| int32 | triton | 96/160 | 15.58 us | `_count_ones_in_block` |
| int32 | triton | 97/160 | 4.48 us | `_count_ones_per_block_blocks` |
| int32 | triton | 98/160 | 4.35 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 99/160 | 4.58 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 100/160 | 45.06 us | `_radix_sort_kernel` |
| int32 | triton | 101/160 | 15.68 us | `_count_ones_in_block` |
| int32 | triton | 102/160 | 4.26 us | `_count_ones_per_block_blocks` |
| int32 | triton | 103/160 | 4.19 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 104/160 | 4.38 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 105/160 | 45.31 us | `_radix_sort_kernel` |
| int32 | triton | 106/160 | 15.58 us | `_count_ones_in_block` |
| int32 | triton | 107/160 | 4.26 us | `_count_ones_per_block_blocks` |
| int32 | triton | 108/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 109/160 | 4.42 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 110/160 | 45.28 us | `_radix_sort_kernel` |
| int32 | triton | 111/160 | 15.55 us | `_count_ones_in_block` |
| int32 | triton | 112/160 | 4.26 us | `_count_ones_per_block_blocks` |
| int32 | triton | 113/160 | 4.26 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 114/160 | 4.42 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 115/160 | 45.31 us | `_radix_sort_kernel` |
| int32 | triton | 116/160 | 15.55 us | `_count_ones_in_block` |
| int32 | triton | 117/160 | 4.42 us | `_count_ones_per_block_blocks` |
| int32 | triton | 118/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 119/160 | 4.38 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 120/160 | 45.09 us | `_radix_sort_kernel` |
| int32 | triton | 121/160 | 15.65 us | `_count_ones_in_block` |
| int32 | triton | 122/160 | 4.48 us | `_count_ones_per_block_blocks` |
| int32 | triton | 123/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 124/160 | 4.45 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 125/160 | 45.22 us | `_radix_sort_kernel` |
| int32 | triton | 126/160 | 15.74 us | `_count_ones_in_block` |
| int32 | triton | 127/160 | 4.32 us | `_count_ones_per_block_blocks` |
| int32 | triton | 128/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 129/160 | 4.42 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 130/160 | 45.70 us | `_radix_sort_kernel` |
| int32 | triton | 131/160 | 16.26 us | `_count_ones_in_block` |
| int32 | triton | 132/160 | 4.32 us | `_count_ones_per_block_blocks` |
| int32 | triton | 133/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 134/160 | 4.67 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 135/160 | 45.22 us | `_radix_sort_kernel` |
| int32 | triton | 136/160 | 15.65 us | `_count_ones_in_block` |
| int32 | triton | 137/160 | 4.51 us | `_count_ones_per_block_blocks` |
| int32 | triton | 138/160 | 4.16 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 139/160 | 4.58 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 140/160 | 45.38 us | `_radix_sort_kernel` |
| int32 | triton | 141/160 | 15.58 us | `_count_ones_in_block` |
| int32 | triton | 142/160 | 4.35 us | `_count_ones_per_block_blocks` |
| int32 | triton | 143/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 144/160 | 4.45 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 145/160 | 45.70 us | `_radix_sort_kernel` |
| int32 | triton | 146/160 | 15.55 us | `_count_ones_in_block` |
| int32 | triton | 147/160 | 4.22 us | `_count_ones_per_block_blocks` |
| int32 | triton | 148/160 | 4.10 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 149/160 | 4.64 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 150/160 | 45.25 us | `_radix_sort_kernel` |
| int32 | triton | 151/160 | 15.55 us | `_count_ones_in_block` |
| int32 | triton | 152/160 | 4.48 us | `_count_ones_per_block_blocks` |
| int32 | triton | 153/160 | 4.32 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 154/160 | 4.67 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 155/160 | 45.31 us | `_radix_sort_kernel` |
| int32 | triton | 156/160 | 15.58 us | `_count_ones_in_block` |
| int32 | triton | 157/160 | 4.61 us | `_count_ones_per_block_blocks` |
| int32 | triton | 158/160 | 4.13 us | `_compute_prefix_sums_per_block_of_blocks` |
| int32 | triton | 159/160 | 4.61 us | `_compute_prefix_sums_per_block` |
| int32 | triton | 160/160 | 44.38 us | `_radix_sort_kernel` |

## Key findings (auto-derived)

- **int32**: Triton is **1.26× faster** (2371.4 µs vs 2978.7 µs).

## NCU's own bottleneck verdict

- **int32 / cutile** — Compute and Memory are well-balanced
- **int32 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_int32.ncu-rep`
- `triton_int32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
