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
| fp16 | triton | 4.19 us | 2.76 % | 0.01 % | 8.08 % | 0.88 % | 1.57 % | — | 64 | 16 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| fp16 | cutile | 3.94 us | 2.95 % | 0.01 % | 7.92 % | 0.93 % | 1.59 % | — | 64 | 16 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| bf16 | triton | 3.97 us | 2.94 % | 0.01 % | 7.97 % | 0.93 % | 1.57 % | — | 64 | 16 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| bf16 | cutile | 3.97 us | 2.94 % | 0.01 % | 7.96 % | 0.93 % | 1.52 % | — | 64 | 16 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| fp32 | triton | 4.06 us | 2.90 % | 0.01 % | 7.95 % | 0.91 % | 1.57 % | — | 64 | 16 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| fp32 | cutile | 3.94 us | 2.96 % | 0.01 % | 7.95 % | 1.02 % | 1.57 % | — | 64 | 16 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| int8 | triton | 8.99 us | 30.57 % | 30.57 % | 19.62 % | 20.50 % | 41.77 % | 2.34 Tbyte/s | 128 | 28 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| int8 | cutile | 8.99 us | 30.57 % | 30.57 % | 19.30 % | 20.46 % | 42.10 % | 2.34 Tbyte/s | 128 | 28 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.06× faster** (3.9 µs vs 4.2 µs).
- **bf16**: cuTile is **1.00× faster** (4.0 µs vs 4.0 µs).
- **fp32**: cuTile is **1.03× faster** (3.9 µs vs 4.1 µs).
- **int8**: cuTile is **1.00× faster** (9.0 µs vs 9.0 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.14 full waves across all SMs. Look at Launch Statistics for more details.
- **bf16 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.14 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.14 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.14 full waves across all SMs. Look at Launch Statistics for more details.
- **fp32 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.14 full waves across all SMs. Look at Launch Statistics for more details.
- **fp32 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.14 full waves across all SMs. Look at Launch Statistics for more details.
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
