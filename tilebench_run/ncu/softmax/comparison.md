# NCU Comparison: softmax

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n_rows': 2048, 'n_cols': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4}` | `{'block_size': 2048, 'occupancy': 8}` |
| fp32 | `{'n_rows': 2048, 'n_cols': 10240}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'block_size': 2048, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 23.78 us | 28.61 % | 24.26 % | 35.80 % | 22.03 % | 63.57 % | 1.86 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp16 | cutile | 43.23 us | 19.68 % | 13.37 % | 23.97 % | 12.56 % | 58.25 % | 1.02 Tbyte/s | 256 | 32 register/thread | 26.83 Kbyte/block | 0 byte/block | 8 block / 8 block |
| fp32 | triton | 32.64 us | 51.24 % | 51.24 % | 53.19 % | 33.76 % | 53.04 % | 3.93 Tbyte/s | 256 | 26 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp32 | cutile | 44.51 us | 35.01 % | 35.01 % | 38.23 % | 23.73 % | 57.88 % | 2.69 Tbyte/s | 256 | 32 register/thread | 18.84 Kbyte/block | 0 byte/block | 8 block / 8 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.82× faster** (23.8 µs vs 43.2 µs).
- **fp32**: Triton is **1.36× faster** (32.6 µs vs 44.5 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — Compute is more heavily utilized than Memory
- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
