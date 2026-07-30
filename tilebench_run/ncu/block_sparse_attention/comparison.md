# NCU Comparison: block_sparse_attention

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'B': 2, 'H': 8, 'M': 10240, 'D': 128, 'H_kv': 2, 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 128}` | `{'num_warps': 4, 'num_stages': 2}` | `{'occupancy': 32}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 92.83 us | 13.76 % | 10.63 % | 13.00 % | 13.74 % | 34.88 % | 815.02 Gbyte/s | 128 | 150 register/thread | 0 byte/block | 81.98 Kbyte/block | 3 block / 2 block |
| fp16 | cutile | 301.15 us | 31.58 % | 3.29 % | 33.51 % | 4.28 % | 23.33 % | 252.39 Gbyte/s | 384 | 168 register/thread | 207.72 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Key findings (auto-derived)

- **fp16**: Triton is **3.24× faster** (92.8 µs vs 301.1 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

## Reports

- `cutile_fp16.ncu-rep`
- `triton_fp16.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
