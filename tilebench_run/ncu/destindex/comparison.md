# NCU Comparison: destindex

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch_size': 1, 'kv_nope_head_num': 12, 'kv_rope_head_num': 1, 'kv_nope_head_dim': 128, 'kv_rope_head_dim': 64, 'seq_len': 40960}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'nope_block_size': 1024, 'nope_occupancy': 4, 'rope_block_size': 512, 'rope_occupancy': 8}` |
| bf16 | `{'batch_size': 1, 'kv_nope_head_num': 12, 'kv_rope_head_num': 1, 'kv_nope_head_dim': 128, 'kv_rope_head_dim': 64, 'seq_len': 40960}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'nope_block_size': 1024, 'nope_occupancy': 4, 'rope_block_size': 512, 'rope_occupancy': 4}` |
| fp32 | `{'batch_size': 1, 'kv_nope_head_num': 12, 'kv_rope_head_num': 1, 'kv_nope_head_dim': 128, 'kv_rope_head_dim': 64, 'seq_len': 40960}` | `{'BLOCK_SIZE': 1024, 'num_warps': 8}` | `{'nope_block_size': 512, 'nope_occupancy': 4, 'rope_block_size': 512, 'rope_occupancy': 16}` |
| int8 | `{'batch_size': 1, 'kv_nope_head_num': 12, 'kv_rope_head_num': 1, 'kv_nope_head_dim': 128, 'kv_rope_head_dim': 64, 'seq_len': 40960}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'nope_block_size': 1024, 'nope_occupancy': 4, 'rope_block_size': 512, 'rope_occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 49.41 us | 78.00 % | 78.00 % | 40.86 % | 35.26 % | 16.13 % | 5.98 Tbyte/s | 64 | 22 register/thread | 0 byte/block | 0 byte/block | 42 block / 32 block |
| fp16 | cutile | 117.24 us | 32.08 % | 32.08 % | 28.79 % | 14.17 % | 70.25 % | 2.46 Tbyte/s | 128 | 40 register/thread | 2.06 Kbyte/block | 0 byte/block | 12 block / 32 block |
| bf16 | triton | 49.15 us | 78.20 % | 78.20 % | 40.07 % | 35.34 % | 16.13 % | 5.99 Tbyte/s | 64 | 22 register/thread | 0 byte/block | 0 byte/block | 42 block / 32 block |
| bf16 | cutile | 116.90 us | 32.21 % | 32.21 % | 28.70 % | 14.22 % | 70.42 % | 2.47 Tbyte/s | 128 | 40 register/thread | 2.06 Kbyte/block | 0 byte/block | 12 block / 32 block |
| fp32 | triton | 93.35 us | 78.50 % | 78.50 % | 37.59 % | 35.44 % | 20.53 % | 6.02 Tbyte/s | 256 | 16 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | cutile | 128.48 us | 57.08 % | 57.08 % | 35.92 % | 25.74 % | 74.82 % | 4.38 Tbyte/s | 128 | 32 register/thread | 2.06 Kbyte/block | 0 byte/block | 16 block / 32 block |
| int8 | triton | 42.84 us | 48.70 % | 48.70 % | 24.79 % | 21.31 % | 11.90 % | 3.73 Tbyte/s | 64 | 16 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| int8 | cutile | 113.05 us | 25.17 % | 17.61 % | 26.17 % | 8.02 % | 75.46 % | 1.35 Tbyte/s | 128 | 40 register/thread | 1.04 Kbyte/block | 0 byte/block | 12 block / 30 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| bf16 | cutile | 1/2 | 108.13 us | `copy_by_dest_kernel_Kt1_A1bf16_1i16t1_p16_A1i64_1i` |
| bf16 | cutile | 2/2 | 8.77 us | `copy_by_dest_kernel_Kt1_A1bf16_1i16t1_p16_A1i64_1i` |
| bf16 | triton | 1/2 | 42.88 us | `copy_by_dest_kernel` |
| bf16 | triton | 2/2 | 6.27 us | `copy_by_dest_kernel` |
| fp16 | cutile | 1/2 | 108.54 us | `copy_by_dest_kernel_Kt1_A1f16_1i16t1_p16_A1i64_1i1` |
| fp16 | cutile | 2/2 | 8.70 us | `copy_by_dest_kernel_Kt1_A1f16_1i16t1_p16_A1i64_1i1` |
| fp16 | triton | 1/2 | 43.01 us | `copy_by_dest_kernel` |
| fp16 | triton | 2/2 | 6.40 us | `copy_by_dest_kernel` |
| fp32 | cutile | 1/2 | 118.02 us | `copy_by_dest_kernel_Kt1_A1f32_1i16t1_p16_A1i64_1i1` |
| fp32 | cutile | 2/2 | 10.46 us | `copy_by_dest_kernel_Kt1_A1f32_1i16t1_p16_A1i64_1i1` |
| fp32 | triton | 1/2 | 85.54 us | `copy_by_dest_kernel` |
| fp32 | triton | 2/2 | 7.81 us | `copy_by_dest_kernel` |
| int8 | cutile | 1/2 | 104.35 us | `copy_by_dest_kernel_Kt1_A1i8_1i16t1_p16_A1i64_1i16` |
| int8 | cutile | 2/2 | 8.70 us | `copy_by_dest_kernel_Kt1_A1i8_1i16t1_p16_A1i64_1i16` |
| int8 | triton | 1/2 | 36.54 us | `copy_by_dest_kernel` |
| int8 | triton | 2/2 | 6.30 us | `copy_by_dest_kernel` |

## Key findings (auto-derived)

- **fp16**: Triton is **2.37× faster** (49.4 µs vs 117.2 µs).
- **bf16**: Triton is **2.38× faster** (49.1 µs vs 116.9 µs).
- **fp32**: Triton is **1.38× faster** (93.3 µs vs 128.5 µs).
- **int8**: Triton is **2.64× faster** (42.8 µs vs 113.0 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Compute is more heavily utilized than Memory
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Compute is more heavily utilized than Memory
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — Compute is more heavily utilized than Memory
- **fp32 / triton** — Memory is more heavily utilized than Compute
- **int8 / cutile** — Compute is more heavily utilized than Memory
- **int8 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `cutile_int8.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`
- `triton_int8.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
