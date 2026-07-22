# NCU Comparison: interleave

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 20000000}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 4096, 'occupancy': 8}` |
| bf16 | `{'n': 20000000}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 2}` | `{'tile': 1024, 'occupancy': 4}` |
| fp32 | `{'n': 20000000}` | `{'BLOCK_SIZE': 1024, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 2048, 'occupancy': 8}` |
| int8 | `{'n': 20000000}` | `{'BLOCK_SIZE': 4096, 'num_warps': 4, 'num_stages': 2}` | `{'tile': 2048, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 21.63 us | 65.64 % | 65.64 % | 64.12 % | 40.35 % | 19.92 % | 5.03 Tbyte/s | 128 | 25 register/thread | 0 byte/block | 4.10 Kbyte/block | 16 block / 26 block |
| fp16 | cutile | 22.69 us | 62.45 % | 62.45 % | 61.12 % | 38.07 % | 25.42 % | 4.79 Tbyte/s | 128 | 62 register/thread | 16.40 Kbyte/block | 0 byte/block | 8 block / 9 block |
| bf16 | triton | 21.47 us | 65.97 % | 65.97 % | 63.37 % | 40.65 % | 19.81 % | 5.06 Tbyte/s | 128 | 25 register/thread | 0 byte/block | 4.10 Kbyte/block | 16 block / 26 block |
| bf16 | cutile | 22.75 us | 63.38 % | 63.38 % | 63.78 % | 38.47 % | 40.10 % | 4.86 Tbyte/s | 128 | 25 register/thread | 4.11 Kbyte/block | 0 byte/block | 16 block / 25 block |
| fp32 | triton | 44.67 us | 77.89 % | 77.89 % | 54.30 % | 40.25 % | 19.82 % | 5.97 Tbyte/s | 256 | 22 register/thread | 0 byte/block | 8.19 Kbyte/block | 10 block / 14 block |
| fp32 | cutile | 45.12 us | 76.52 % | 76.52 % | 55.31 % | 39.81 % | 19.37 % | 5.86 Tbyte/s | 128 | 52 register/thread | 16.40 Kbyte/block | 0 byte/block | 9 block / 9 block |
| int8 | triton | 13.12 us | 42.06 % | 41.77 % | 64.49 % | 33.02 % | 16.70 % | 3.20 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 4.10 Kbyte/block | 16 block / 26 block |
| int8 | cutile | 13.06 us | 43.67 % | 42.30 % | 62.97 % | 34.09 % | 33.92 % | 3.23 Tbyte/s | 128 | 23 register/thread | 4.11 Kbyte/block | 0 byte/block | 21 block / 25 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.05× faster** (21.6 µs vs 22.7 µs).
- **bf16**: Triton is **1.06× faster** (21.5 µs vs 22.8 µs).
- **fp32**: Triton is **1.01× faster** (44.7 µs vs 45.1 µs).
- **int8**: cuTile is **1.00× faster** (13.1 µs vs 13.1 µs).

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
