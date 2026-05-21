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
| fp32 | triton | 19.23 us | 43.16 % | 30.86 % | 55.11 % | 34.67 % | 40.54 % | 2.36 Tbyte/s | 512 | 32 register/thread | 0 byte/block | 16.38 Kbyte/block | 4 block / 7 block |
| fp32 | cutile | 21.18 us | 39.15 % | 28.21 % | 50.26 % | 33.30 % | 35.21 % | 2.16 Tbyte/s | 128 | 64 register/thread | 16.40 Kbyte/block | 0 byte/block | 8 block / 9 block |

## Key findings (auto-derived)

- **fp32**: Triton is **1.10× faster** (19.2 µs vs 21.2 µs).

## NCU's own bottleneck verdict

- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

## Reports

- `cutile_fp32.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
