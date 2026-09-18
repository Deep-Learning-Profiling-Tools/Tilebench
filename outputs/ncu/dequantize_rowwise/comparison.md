# NCU Comparison: dequantize_rowwise

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| int8 | `{'rows': 4096, 'cols': 8192}` | `{'CHUNK': 1024, 'num_warps': 2}` | `{'chunk': 1024, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| int8 | triton | 22.21 us | 58.40 % | 58.40 % | 43.19 % | 30.22 % | 32.80 % | 4.47 Tbyte/s | 64 | 30 register/thread | 0 byte/block | 2.05 Kbyte/block | 32 block / 33 block |
| int8 | cutile | 26.43 us | 49.59 % | 49.59 % | 35.92 % | 25.35 % | 50.32 % | 3.79 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |

## Key findings (auto-derived)

- **int8**: Triton is **1.19× faster** (22.2 µs vs 26.4 µs).

## NCU's own bottleneck verdict

- **int8 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **int8 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

## Reports

- `cutile_int8.ncu-rep`
- `triton_int8.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
