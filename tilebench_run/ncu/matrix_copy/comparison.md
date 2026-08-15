# NCU Comparison: matrix_copy

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'N': 5120}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 1024, 'occupancy': 8}` |
| bf16 | `{'N': 5120}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 2}` | `{'tile': 1024, 'occupancy': 16}` |
| fp32 | `{'N': 5120}` | `{'BLOCK_SIZE': 4096, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 1024, 'occupancy': 16}` |
| int8 | `{'N': 5120}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4, 'num_stages': 2}` | `{'tile': 2048, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 21.50 us | 65.04 % | 65.04 % | 35.48 % | 29.46 % | 13.55 % | 4.98 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| fp16 | cutile | 22.43 us | 63.81 % | 63.81 % | 35.10 % | 28.51 % | 24.11 % | 4.88 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | triton | 21.98 us | 63.83 % | 63.83 % | 35.37 % | 28.87 % | 13.29 % | 4.89 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | cutile | 21.82 us | 64.31 % | 64.31 % | 35.06 % | 29.08 % | 24.09 % | 4.93 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| fp32 | triton | 37.15 us | 73.90 % | 73.90 % | 40.99 % | 33.80 % | 7.85 % | 5.66 Tbyte/s | 128 | 45 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 36.03 us | 76.31 % | 76.31 % | 39.95 % | 34.88 % | 21.25 % | 5.85 Tbyte/s | 128 | 18 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| int8 | triton | 13.41 us | 50.32 % | 50.32 % | 32.54 % | 23.47 % | 7.46 % | 3.85 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| int8 | cutile | 13.86 us | 49.78 % | 49.78 % | 31.26 % | 22.88 % | 19.46 % | 3.80 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.04× faster** (21.5 µs vs 22.4 µs).
- **bf16**: cuTile is **1.01× faster** (21.8 µs vs 22.0 µs).
- **fp32**: cuTile is **1.03× faster** (36.0 µs vs 37.1 µs).
- **int8**: Triton is **1.03× faster** (13.4 µs vs 13.9 µs).

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
