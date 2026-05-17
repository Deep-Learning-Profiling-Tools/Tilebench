# NCU Comparison: destindex

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch_size': 1, 'kv_nope_head_num': 12, 'kv_rope_head_num': 1, 'kv_nope_head_dim': 128, 'kv_rope_head_dim': 64, 'seq_len': 40960}` | `{'BLOCK_DMODEL': 128, 'num_warps': 1, 'num_stages': 1}` | `{'nope': {'block_d': 64, 'occupancy': 8}, 'rope': {'block_d': 64, 'occupancy': 16}}` |
| bf16 | `{'batch_size': 1, 'kv_nope_head_num': 12, 'kv_rope_head_num': 1, 'kv_nope_head_dim': 128, 'kv_rope_head_dim': 64, 'seq_len': 40960}` | `{'BLOCK_DMODEL': 128, 'num_warps': 2, 'num_stages': 1}` | `{'nope': {'block_d': 64, 'occupancy': 8}, 'rope': {'block_d': 64, 'occupancy': 16}}` |
| fp32 | `{'batch_size': 1, 'kv_nope_head_num': 12, 'kv_rope_head_num': 1, 'kv_nope_head_dim': 128, 'kv_rope_head_dim': 64, 'seq_len': 40960}` | `{'BLOCK_DMODEL': 64, 'num_warps': 1, 'num_stages': 1}` | `{'nope': {'block_d': 64, 'occupancy': 8}, 'rope': {'block_d': 64, 'occupancy': 16}}` |
| int8 | `{'batch_size': 1, 'kv_nope_head_num': 12, 'kv_rope_head_num': 1, 'kv_nope_head_dim': 128, 'kv_rope_head_dim': 64, 'seq_len': 40960}` | `{'BLOCK_DMODEL': 128, 'num_warps': 2, 'num_stages': 2}` | `{'nope': {'block_d': 64, 'occupancy': 8}, 'rope': {'block_d': 64, 'occupancy': 16}}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 16.60 us | 1.65 % | 0.74 % | 41.46 % | 0.91 % | 0.88 % | 56.63 Gbyte/s | 384 | 80 register/thread | 38.91 Kbyte/block | 0 byte/block | 2 block / 3 block |
| fp16 | cutile | 16.29 us | 1.68 % | 0.75 % | 42.20 % | 0.92 % | 0.89 % | 57.82 Gbyte/s | 384 | 80 register/thread | 38.91 Kbyte/block | 0 byte/block | 2 block / 3 block |
| bf16 | triton | 16.64 us | 1.68 % | 0.75 % | 42.13 % | 0.92 % | 0.90 % | 57.23 Gbyte/s | 384 | 80 register/thread | 38.91 Kbyte/block | 0 byte/block | 2 block / 3 block |
| bf16 | cutile | 16.41 us | 1.69 % | 0.75 % | 41.50 % | 0.91 % | 0.90 % | 57.40 Gbyte/s | 384 | 80 register/thread | 38.91 Kbyte/block | 0 byte/block | 2 block / 3 block |
| fp32 | triton | 16.80 us | 1.67 % | 0.74 % | 40.10 % | 0.90 % | 0.89 % | 56.48 Gbyte/s | 384 | 80 register/thread | 38.91 Kbyte/block | 0 byte/block | 2 block / 3 block |
| fp32 | cutile | 16.73 us | 1.63 % | 0.74 % | 41.35 % | 0.91 % | 0.87 % | 56.63 Gbyte/s | 384 | 80 register/thread | 38.91 Kbyte/block | 0 byte/block | 2 block / 3 block |
| int8 | triton | 10.82 us | 6.93 % | 0.78 % | 25.29 % | 0.59 % | 14.38 % | 59.65 Gbyte/s | 128 | 64 register/thread | 8.19 Kbyte/block | 0 byte/block | 8 block / 14 block |
| int8 | cutile | 10.82 us | 7.19 % | 0.78 % | 24.73 % | 0.59 % | 14.89 % | 59.24 Gbyte/s | 128 | 64 register/thread | 8.19 Kbyte/block | 0 byte/block | 8 block / 14 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| bf16 | cutile | 1/2 | 4.35 us | `void DeviceRadixSortExclusiveSumKernel<policy_hub<` |
| bf16 | cutile | 2/2 | 12.06 us | `void DeviceRadixSortOnesweepKernel<policy_hub<long` |
| bf16 | triton | 1/2 | 4.54 us | `void DeviceRadixSortExclusiveSumKernel<policy_hub<` |
| bf16 | triton | 2/2 | 12.10 us | `void DeviceRadixSortOnesweepKernel<policy_hub<long` |
| fp16 | cutile | 1/2 | 4.32 us | `void DeviceRadixSortExclusiveSumKernel<policy_hub<` |
| fp16 | cutile | 2/2 | 11.97 us | `void DeviceRadixSortOnesweepKernel<policy_hub<long` |
| fp16 | triton | 1/2 | 4.38 us | `void DeviceRadixSortExclusiveSumKernel<policy_hub<` |
| fp16 | triton | 2/2 | 12.22 us | `void DeviceRadixSortOnesweepKernel<policy_hub<long` |
| fp32 | cutile | 1/2 | 4.51 us | `void DeviceRadixSortExclusiveSumKernel<policy_hub<` |
| fp32 | cutile | 2/2 | 12.22 us | `void DeviceRadixSortOnesweepKernel<policy_hub<long` |
| fp32 | triton | 1/2 | 4.54 us | `void DeviceRadixSortExclusiveSumKernel<policy_hub<` |
| fp32 | triton | 2/2 | 12.26 us | `void DeviceRadixSortOnesweepKernel<policy_hub<long` |
| int8 | cutile | 1/2 | 4.90 us | `void native::<unnamed>::distribution_elementwise_g` |
| int8 | cutile | 2/2 | 5.92 us | `void at_cuda_detail::DeviceRadixSortHistogramKerne` |
| int8 | triton | 1/2 | 4.93 us | `void native::<unnamed>::distribution_elementwise_g` |
| int8 | triton | 2/2 | 5.89 us | `void at_cuda_detail::DeviceRadixSortHistogramKerne` |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.02× faster** (16.3 µs vs 16.6 µs).
- **bf16**: cuTile is **1.01× faster** (16.4 µs vs 16.6 µs).
- **fp32**: cuTile is **1.00× faster** (16.7 µs vs 16.8 µs).
- **int8**: cuTile is **1.00× faster** (10.8 µs vs 10.8 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.03 full waves across all SMs. Look at Launch Statistics for more details.
- **bf16 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.03 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.03 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.03 full waves across all SMs. Look at Launch Statistics for more details.
- **fp32 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.03 full waves across all SMs. Look at Launch Statistics for more details.
- **fp32 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.03 full waves across all SMs. Look at Launch Statistics for more details.
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
