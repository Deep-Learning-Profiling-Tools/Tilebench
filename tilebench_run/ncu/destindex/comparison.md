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
| fp16 | triton | 42.47 us | 70.97 % | 70.97 % | 47.49 % | 37.99 % | 18.48 % | 5.44 Tbyte/s | 64 | 22 register/thread | 0 byte/block | 0 byte/block | 42 block / 32 block |
| fp16 | cutile | 114.95 us | 28.22 % | 25.78 % | 29.46 % | 13.27 % | 71.52 % | 1.98 Tbyte/s | 128 | 40 register/thread | 2.06 Kbyte/block | 0 byte/block | 12 block / 32 block |
| bf16 | triton | 42.37 us | 71.40 % | 71.40 % | 47.49 % | 38.13 % | 18.48 % | 5.47 Tbyte/s | 64 | 22 register/thread | 0 byte/block | 0 byte/block | 42 block / 32 block |
| bf16 | cutile | 115.27 us | 28.27 % | 25.80 % | 29.40 % | 13.30 % | 71.65 % | 1.98 Tbyte/s | 128 | 40 register/thread | 2.06 Kbyte/block | 0 byte/block | 12 block / 32 block |
| fp32 | triton | 85.22 us | 76.17 % | 76.17 % | 42.01 % | 37.11 % | 22.46 % | 5.84 Tbyte/s | 256 | 16 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | cutile | 126.59 us | 51.19 % | 51.19 % | 36.42 % | 24.89 % | 75.65 % | 3.93 Tbyte/s | 128 | 32 register/thread | 2.06 Kbyte/block | 0 byte/block | 16 block / 32 block |
| int8 | triton | 41.47 us | 30.41 % | 30.41 % | 25.27 % | 19.68 % | 12.12 % | 2.33 Tbyte/s | 64 | 16 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| int8 | cutile | 111.61 us | 25.38 % | 10.89 % | 26.48 % | 7.62 % | 76.09 % | 835.22 Gbyte/s | 128 | 40 register/thread | 1.04 Kbyte/block | 0 byte/block | 12 block / 30 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| bf16 | cutile | 1/2 | 106.69 us | `copy_by_dest_kernel_Kt1_A1bf16_1i16t1_p16_A1i64_1i` |
| bf16 | cutile | 2/2 | 8.58 us | `copy_by_dest_kernel_Kt1_A1bf16_1i16t1_p16_A1i64_1i` |
| bf16 | triton | 1/2 | 36.77 us | `copy_by_dest_kernel` |
| bf16 | triton | 2/2 | 5.60 us | `copy_by_dest_kernel` |
| fp16 | cutile | 1/2 | 106.50 us | `copy_by_dest_kernel_Kt1_A1f16_1i16t1_p16_A1i64_1i1` |
| fp16 | cutile | 2/2 | 8.45 us | `copy_by_dest_kernel_Kt1_A1f16_1i16t1_p16_A1i64_1i1` |
| fp16 | triton | 1/2 | 36.90 us | `copy_by_dest_kernel` |
| fp16 | triton | 2/2 | 5.57 us | `copy_by_dest_kernel` |
| fp32 | cutile | 1/2 | 117.25 us | `copy_by_dest_kernel_Kt1_A1f32_1i16t1_p16_A1i64_1i1` |
| fp32 | cutile | 2/2 | 9.34 us | `copy_by_dest_kernel_Kt1_A1f32_1i16t1_p16_A1i64_1i1` |
| fp32 | triton | 1/2 | 78.50 us | `copy_by_dest_kernel` |
| fp32 | triton | 2/2 | 6.72 us | `copy_by_dest_kernel` |
| int8 | cutile | 1/2 | 103.07 us | `copy_by_dest_kernel_Kt1_A1i8_1i16t1_p16_A1i64_1i16` |
| int8 | cutile | 2/2 | 8.54 us | `copy_by_dest_kernel_Kt1_A1i8_1i16t1_p16_A1i64_1i16` |
| int8 | triton | 1/2 | 35.74 us | `copy_by_dest_kernel` |
| int8 | triton | 2/2 | 5.73 us | `copy_by_dest_kernel` |

## Key findings (auto-derived)

- **fp16**: Triton is **2.71× faster** (42.5 µs vs 115.0 µs).
- **bf16**: Triton is **2.72× faster** (42.4 µs vs 115.3 µs).
- **fp32**: Triton is **1.49× faster** (85.2 µs vs 126.6 µs).
- **int8**: Triton is **2.69× faster** (41.5 µs vs 111.6 µs).

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
