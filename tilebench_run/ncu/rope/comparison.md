# NCU Comparison: rope

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch_size': 1, 'n_heads': 32, 'head_dim': 128, 'seq_len': 20480}` | `{'ROPE_GROUP_SIZE': 8, 'num_warps': 2, 'num_stages': 2}` | `{'group_size': 16, 'occupancy': 8}` |
| fp32 | `{'batch_size': 1, 'n_heads': 32, 'head_dim': 128, 'seq_len': 20480}` | `{'ROPE_GROUP_SIZE': 4, 'num_warps': 2, 'num_stages': 3}` | `{'group_size': 16, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 90.78 us | 40.82 % | 40.82 % | 28.05 % | 21.70 % | 25.87 % | 3.13 Tbyte/s | 64 | 20 register/thread | 0 byte/block | 0 byte/block | 42 block / 32 block |
| fp16 | cutile | 201.70 us | 33.77 % | 18.49 % | 34.70 % | 11.53 % | 73.30 % | 1.42 Tbyte/s | 128 | 29 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | triton | 131.81 us | 62.47 % | 62.47 % | 35.84 % | 31.74 % | 22.49 % | 4.79 Tbyte/s | 64 | 24 register/thread | 0 byte/block | 0 byte/block | 42 block / 32 block |
| fp32 | cutile | 212.38 us | 38.84 % | 38.84 % | 33.70 % | 19.40 % | 58.71 % | 2.98 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **2.22× faster** (90.8 µs vs 201.7 µs).
- **fp32**: Triton is **1.61× faster** (131.8 µs vs 212.4 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — Compute is more heavily utilized than Memory
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
