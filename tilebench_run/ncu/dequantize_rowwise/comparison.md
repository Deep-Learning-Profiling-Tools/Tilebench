# NCU Comparison: dequantize_rowwise

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'rows': 4096, 'cols': 8192}` | `{'num_warps': 16}` | `{'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 18.43 us | 42.69 % | 32.19 % | 54.70 % | 36.09 % | 40.10 % | 2.47 Tbyte/s | 512 | 32 register/thread | 0 byte/block | 16.38 Kbyte/block | 4 block / 7 block |
| fp32 | cutile | 21.50 us | 40.59 % | 27.95 % | 51.08 % | 32.76 % | 36.51 % | 2.14 Tbyte/s | 128 | 64 register/thread | 16.40 Kbyte/block | 0 byte/block | 8 block / 9 block |

## Key findings (auto-derived)

- **fp32**: Triton is **1.17× faster** (18.4 µs vs 21.5 µs).

## NCU's own bottleneck verdict

- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

## Reports

- `cutile_fp32.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
