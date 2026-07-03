# NCU Comparison: flash_attention

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 20480}` | `{'BLOCK_M': 128, 'BLOCK_N': 32, 'num_warps': 8, 'num_stages': 4}` | `{'tile_m': 128, 'tile_n': 128, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 22770.00 us | 21.12 % | 1.54 % | 41.56 % | 20.57 % | 60.94 % | 117.85 Gbyte/s | 256 | 128 register/thread | 0 byte/block | 98.85 Kbyte/block | 2 block / 2 block |
| fp16 | cutile | 17670.00 us | 26.41 % | 1.97 % | 26.72 % | 26.40 % | 43.21 % | 151.33 Gbyte/s | 384 | 168 register/thread | 229.74 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.29× faster** (17670.0 µs vs 22770.0 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — Compute is more heavily utilized than Memory

## Reports

- `cutile_fp16.ncu-rep`
- `triton_fp16.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
