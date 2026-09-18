# NCU Comparison: mul2

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'tile': 1024, 'occupancy': 4}` |
| bf16 | `{'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 2}` | `{'tile': 1024, 'occupancy': 32}` |
| fp32 | `{'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'tile': 2048, 'occupancy': 8}` |
| int8 | `{'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 2}` | `{'tile': 2048, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 18.08 us | 59.57 % | 59.57 % | 39.19 % | 27.67 % | 6.84 % | 4.56 Tbyte/s | 64 | 18 register/thread | 0 byte/block | 0 byte/block | 42 block / 32 block |
| fp16 | cutile | 19.65 us | 56.25 % | 56.25 % | 34.06 % | 25.69 % | 25.64 % | 4.30 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | triton | 17.89 us | 60.85 % | 60.85 % | 40.72 % | 28.21 % | 6.43 % | 4.65 Tbyte/s | 64 | 26 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | cutile | 18.88 us | 59.45 % | 59.45 % | 34.73 % | 26.95 % | 25.43 % | 4.55 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| fp32 | triton | 29.54 us | 74.51 % | 74.51 % | 41.02 % | 34.08 % | 7.90 % | 5.71 Tbyte/s | 256 | 18 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 30.27 us | 72.56 % | 72.56 % | 40.26 % | 33.37 % | 17.92 % | 5.55 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| int8 | triton | 11.26 us | 46.58 % | 46.58 % | 34.40 % | 22.28 % | 30.49 % | 3.56 Tbyte/s | 64 | 29 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| int8 | cutile | 12.93 us | 40.79 % | 40.79 % | 27.93 % | 19.30 % | 34.17 % | 3.12 Tbyte/s | 128 | 20 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.09× faster** (18.1 µs vs 19.6 µs).
- **bf16**: Triton is **1.06× faster** (17.9 µs vs 18.9 µs).
- **fp32**: Triton is **1.02× faster** (29.5 µs vs 30.3 µs).
- **int8**: Triton is **1.15× faster** (11.3 µs vs 12.9 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — Memory is more heavily utilized than Compute
- **fp32 / triton** — Memory is more heavily utilized than Compute
- **int8 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **int8 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `cutile_int8.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`
- `triton_int8.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
