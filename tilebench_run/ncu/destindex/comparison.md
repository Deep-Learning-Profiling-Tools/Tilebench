# NCU Comparison: destindex

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch_size': 1, 'kv_nope_head_num': 12, 'kv_rope_head_num': 1, 'kv_nope_head_dim': 128, 'kv_rope_head_dim': 64, 'seq_len': 40960}` | `{'BLOCK_DMODEL': 64, 'num_warps': 1, 'num_stages': 1}` | `{'nope_block_d': 128, 'nope_occupancy': 16, 'rope_block_d': 64, 'rope_occupancy': 16}` |
| bf16 | `{'batch_size': 1, 'kv_nope_head_num': 12, 'kv_rope_head_num': 1, 'kv_nope_head_dim': 128, 'kv_rope_head_dim': 64, 'seq_len': 40960}` | `{'BLOCK_DMODEL': 64, 'num_warps': 1, 'num_stages': 1}` | `{'nope_block_d': 128, 'nope_occupancy': 16, 'rope_block_d': 64, 'rope_occupancy': 16}` |
| fp32 | `{'batch_size': 1, 'kv_nope_head_num': 12, 'kv_rope_head_num': 1, 'kv_nope_head_dim': 128, 'kv_rope_head_dim': 64, 'seq_len': 40960}` | `{'BLOCK_DMODEL': 64, 'num_warps': 2, 'num_stages': 1}` | `{'nope_block_d': 128, 'nope_occupancy': 16, 'rope_block_d': 64, 'rope_occupancy': 16}` |
| int8 | `{'batch_size': 1, 'kv_nope_head_num': 12, 'kv_rope_head_num': 1, 'kv_nope_head_dim': 128, 'kv_rope_head_dim': 64, 'seq_len': 40960}` | `{'BLOCK_DMODEL': 64, 'num_warps': 1, 'num_stages': 1}` | `{'nope_block_d': 128, 'nope_occupancy': 16, 'rope_block_d': 64, 'rope_occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 282.75 us | 10.63 % | 10.63 % | 7.29 % | 5.72 % | 11.51 % | 815.61 Gbyte/s | 32 | 22 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| fp16 | cutile | 282.59 us | 10.95 % | 10.71 % | 11.17 % | 6.32 % | 47.64 % | 821.33 Gbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| bf16 | triton | 282.62 us | 10.62 % | 10.62 % | 7.29 % | 5.73 % | 11.48 % | 814.84 Gbyte/s | 32 | 22 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| bf16 | cutile | 282.24 us | 10.97 % | 10.71 % | 11.17 % | 6.33 % | 47.73 % | 821.77 Gbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| fp32 | triton | 283.17 us | 23.35 % | 23.35 % | 13.34 % | 11.38 % | 22.75 % | 1.79 Tbyte/s | 64 | 24 register/thread | 0 byte/block | 0 byte/block | 42 block / 32 block |
| fp32 | cutile | 283.17 us | 23.36 % | 23.36 % | 13.16 % | 11.38 % | 47.66 % | 1.79 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| int8 | triton | 282.69 us | 4.56 % | 4.42 % | 4.66 % | 3.65 % | 10.51 % | 339.25 Gbyte/s | 32 | 23 register/thread | 0 byte/block | 0 byte/block | 84 block / 32 block |
| int8 | cutile | 282.47 us | 10.90 % | 4.32 % | 11.08 % | 3.63 % | 44.38 % | 331.56 Gbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| bf16 | cutile | 1/2 | 256.83 us | `copy_by_dest_kernel_Kt1_A3bf16_3v8l0_4t1_5i16_p16_` |
| bf16 | cutile | 2/2 | 25.41 us | `copy_by_dest_kernel_Kt1_A3bf16_3v8l0_4t1_5i16_p16_` |
| bf16 | triton | 1/2 | 257.31 us | `copy_by_dest_kernel` |
| bf16 | triton | 2/2 | 25.31 us | `copy_by_dest_kernel` |
| fp16 | cutile | 1/2 | 257.15 us | `copy_by_dest_kernel_Kt1_A3f16_3v8l0_4t1_5i16_p16_A` |
| fp16 | cutile | 2/2 | 25.44 us | `copy_by_dest_kernel_Kt1_A3f16_3v8l0_4t1_5i16_p16_A` |
| fp16 | triton | 1/2 | 257.41 us | `copy_by_dest_kernel` |
| fp16 | triton | 2/2 | 25.34 us | `copy_by_dest_kernel` |
| fp32 | cutile | 1/2 | 257.70 us | `copy_by_dest_kernel_Kt1_A3f32_3v4l0_4t1_5i16_p16_A` |
| fp32 | cutile | 2/2 | 25.47 us | `copy_by_dest_kernel_Kt1_A3f32_3v4l0_4t1_5i16_p16_A` |
| fp32 | triton | 1/2 | 257.63 us | `copy_by_dest_kernel` |
| fp32 | triton | 2/2 | 25.54 us | `copy_by_dest_kernel` |
| int8 | cutile | 1/2 | 257.22 us | `copy_by_dest_kernel_Kt1_A3i8_3v16l0_4t1_5i16_p16_A` |
| int8 | cutile | 2/2 | 25.25 us | `copy_by_dest_kernel_Kt1_A3i8_3v16l0_4t1_5i16_p16_A` |
| int8 | triton | 1/2 | 257.12 us | `copy_by_dest_kernel` |
| int8 | triton | 2/2 | 25.57 us | `copy_by_dest_kernel` |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.00× faster** (282.6 µs vs 282.8 µs).
- **bf16**: cuTile is **1.00× faster** (282.2 µs vs 282.6 µs).
- **fp32**: cuTile is **1.00× faster** (283.2 µs vs 283.2 µs).
- **int8**: cuTile is **1.00× faster** (282.5 µs vs 282.7 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **bf16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **int8 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
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
