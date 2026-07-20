# NCU Comparison: linear_self_attention

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'M': 10000, 'D': 256, 'eps': 1e-06}` | `{'kv_BLOCK_M': 32, 'kv_BLOCK_N': 32, 'kv_BLOCK_K': 64, 'kv_num_warps': 4, 'kv_num_stages': 3, 'out_BLOCK_M': 64, 'out_BLOCK_N': 128, 'out_BLOCK_K': 32, 'out_num_warps': 4, 'out_num_stages': 3}` | `{'kv_block_m': 16, 'kv_block_n': 64, 'kv_block_k': 32, 'kv_occupancy': 4, 'out_block_m': 32, 'out_block_n': 64, 'out_block_k': 32, 'out_occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 63.08 us | 22.82 % | 7.56 % | 36.63 % | 13.09 % | 19.18 % | 578.75 Gbyte/s | 128 | 48 register/thread | 0 byte/block | 49.23 Kbyte/block | 10 block / 4 block |
| fp32 | cutile | 2251.47 us | 2.24 % | 0.15 % | 10.31 % | 0.56 % | 1.37 % | 11.32 Gbyte/s | 128 | 64 register/thread | 8.30 Kbyte/block | 0 byte/block | 8 block / 14 block |

## Key findings (auto-derived)

- **fp32**: Triton is **35.69× faster** (63.1 µs vs 2251.5 µs).

## NCU's own bottleneck verdict

- **fp32 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.03 full waves across all SMs. Look at Launch Statistics for more details.
- **fp32 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

## Reports

- `cutile_fp32.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
