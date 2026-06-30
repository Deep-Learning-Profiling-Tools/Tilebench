# NCU Comparison: interleave

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 20000000}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 8192, 'occupancy': 4}` |
| bf16 | `{'n': 20000000}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 8192, 'occupancy': 4}` |
| fp32 | `{'n': 20000000}` | `{'BLOCK_SIZE': 1024, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 8192, 'occupancy': 4}` |
| int8 | `{'n': 20000000}` | `{'BLOCK_SIZE': 4096, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 8192, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 21.41 us | 66.39 % | 66.39 % | 63.79 % | 40.99 % | 20.23 % | 5.08 Tbyte/s | 128 | 25 register/thread | 0 byte/block | 4.10 Kbyte/block | 16 block / 26 block |
| fp16 | cutile | 22.37 us | 62.64 % | 62.64 % | 60.73 % | 38.33 % | 20.92 % | 4.79 Tbyte/s | 128 | 96 register/thread | 16.40 Kbyte/block | 0 byte/block | 5 block / 7 block |
| bf16 | triton | 21.44 us | 66.27 % | 66.27 % | 64.36 % | 40.64 % | 20.38 % | 5.08 Tbyte/s | 128 | 25 register/thread | 0 byte/block | 4.10 Kbyte/block | 16 block / 26 block |
| bf16 | cutile | 22.50 us | 62.55 % | 62.55 % | 60.51 % | 38.07 % | 20.88 % | 4.79 Tbyte/s | 128 | 96 register/thread | 16.40 Kbyte/block | 0 byte/block | 5 block / 7 block |
| fp32 | triton | 43.94 us | 79.08 % | 79.08 % | 53.56 % | 40.89 % | 19.94 % | 6.06 Tbyte/s | 256 | 22 register/thread | 0 byte/block | 8.19 Kbyte/block | 10 block / 14 block |
| fp32 | cutile | 51.39 us | 70.98 % | 70.98 % | 55.89 % | 42.70 % | 19.72 % | 5.44 Tbyte/s | 128 | 128 register/thread | 16.40 Kbyte/block | 0 byte/block | 4 block / 7 block |
| int8 | triton | 13.18 us | 41.65 % | 41.65 % | 63.53 % | 32.78 % | 16.51 % | 3.18 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 4.10 Kbyte/block | 16 block / 26 block |
| int8 | cutile | 16.86 us | 33.42 % | 33.42 % | 41.27 % | 26.21 % | 40.27 % | 2.56 Tbyte/s | 128 | 86 register/thread | 16.40 Kbyte/block | 0 byte/block | 5 block / 7 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.04× faster** (21.4 µs vs 22.4 µs).
- **bf16**: Triton is **1.05× faster** (21.4 µs vs 22.5 µs).
- **fp32**: Triton is **1.17× faster** (43.9 µs vs 51.4 µs).
- **int8**: Triton is **1.28× faster** (13.2 µs vs 16.9 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Memory is more heavily utilized than Compute
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Memory is more heavily utilized than Compute
- **fp16 / triton** — Memory is more heavily utilized than Compute
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
