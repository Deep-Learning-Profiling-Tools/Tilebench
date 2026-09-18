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
| fp16 | triton | 25.38 us | 47.17 % | 47.17 % | 33.23 % | 23.43 % | 60.04 % | 3.61 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp16 | cutile | 44.19 us | 26.36 % | 26.36 % | 23.16 % | 13.45 % | 56.17 % | 2.02 Tbyte/s | 256 | 32 register/thread | 26.83 Kbyte/block | 0 byte/block | 8 block / 8 block |
| fp32 | triton | 37.92 us | 66.20 % | 66.20 % | 44.12 % | 33.23 % | 46.52 % | 5.08 Tbyte/s | 256 | 26 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp32 | cutile | 47.39 us | 48.19 % | 48.19 % | 35.88 % | 24.69 % | 55.05 % | 3.69 Tbyte/s | 256 | 32 register/thread | 18.84 Kbyte/block | 0 byte/block | 8 block / 8 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.74× faster** (25.4 µs vs 44.2 µs).
- **fp32**: Triton is **1.25× faster** (37.9 µs vs 47.4 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — Compute is more heavily utilized than Memory
- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
