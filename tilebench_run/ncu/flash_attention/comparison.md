# NCU Comparison: flash_attention

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'seq_len': 20480, 'causal': True}` | `{'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 8, 'num_stages': 2}` | `{'tile_m': 128, 'tile_n': 128, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 19410.00 us | 24.06 % | 1.80 % | 32.59 % | 24.08 % | 51.24 % | 138.14 Gbyte/s | 256 | 128 register/thread | 0 byte/block | 98.88 Kbyte/block | 2 block / 2 block |
| fp16 | cutile | 17670.00 us | 26.40 % | 1.97 % | 26.72 % | 26.41 % | 43.20 % | 151.51 Gbyte/s | 384 | 168 register/thread | 229.74 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.10× faster** (17670.0 µs vs 19410.0 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

## Reports

- `cutile_fp16.ncu-rep`
- `triton_fp16.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
