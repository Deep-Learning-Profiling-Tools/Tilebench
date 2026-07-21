# NCU Comparison: interleave

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 20000000}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 2}` | `{'tile': 1024, 'occupancy': 16}` |
| bf16 | `{'n': 20000000}` | `{'BLOCK_SIZE': 4096, 'num_warps': 4, 'num_stages': 2}` | `{'tile': 1024, 'occupancy': 16}` |
| fp32 | `{'n': 20000000}` | `{'BLOCK_SIZE': 1024, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 1024, 'occupancy': 16}` |
| int8 | `{'n': 20000000}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 2048, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 21.86 us | 65.14 % | 65.14 % | 62.78 % | 39.79 % | 20.30 % | 4.99 Tbyte/s | 128 | 25 register/thread | 0 byte/block | 4.10 Kbyte/block | 16 block / 26 block |
| fp16 | cutile | 35.36 us | 64.77 % | 40.20 % | 74.59 % | 54.27 % | 37.01 % | 3.08 Tbyte/s | 128 | 32 register/thread | 4.11 Kbyte/block | 0 byte/block | 16 block / 25 block |
| bf16 | triton | 22.40 us | 62.16 % | 62.16 % | 58.14 % | 37.95 % | 19.16 % | 4.76 Tbyte/s | 128 | 48 register/thread | 0 byte/block | 4.10 Kbyte/block | 10 block / 20 block |
| bf16 | cutile | 35.33 us | 65.36 % | 40.26 % | 74.26 % | 54.26 % | 37.35 % | 3.09 Tbyte/s | 128 | 32 register/thread | 4.11 Kbyte/block | 0 byte/block | 16 block / 25 block |
| fp32 | triton | 44.64 us | 77.71 % | 77.71 % | 53.61 % | 40.26 % | 19.95 % | 5.95 Tbyte/s | 256 | 22 register/thread | 0 byte/block | 8.19 Kbyte/block | 10 block / 14 block |
| fp32 | cutile | 59.74 us | 64.32 % | 58.12 % | 70.36 % | 52.30 % | 30.50 % | 4.46 Tbyte/s | 128 | 32 register/thread | 8.20 Kbyte/block | 0 byte/block | 16 block / 17 block |
| int8 | triton | 12.58 us | 44.79 % | 44.78 % | 64.63 % | 35.29 % | 17.54 % | 3.43 Tbyte/s | 128 | 23 register/thread | 0 byte/block | 4.10 Kbyte/block | 21 block / 26 block |
| int8 | cutile | 24.00 us | 57.29 % | 22.22 % | 69.92 % | 44.91 % | 42.86 % | 1.70 Tbyte/s | 128 | 56 register/thread | 4.11 Kbyte/block | 0 byte/block | 9 block / 19 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.62× faster** (21.9 µs vs 35.4 µs).
- **bf16**: Triton is **1.58× faster** (22.4 µs vs 35.3 µs).
- **fp32**: Triton is **1.34× faster** (44.6 µs vs 59.7 µs).
- **int8**: Triton is **1.91× faster** (12.6 µs vs 24.0 µs).

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
