# NCU Comparison: mean_reduction

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_M': 1, 'BLOCK_N': 1024, 'num_warps': 8}` | `{'tile_size': 512, 'occupancy': 4}` |
| bf16 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_M': 1, 'BLOCK_N': 1024, 'num_warps': 8}` | `{'tile_size': 512, 'occupancy': 4}` |
| fp32 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_M': 1, 'BLOCK_N': 1024, 'num_warps': 8}` | `{'tile_size': 512, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 16.80 us | 66.64 % | 66.64 % | 24.84 % | 43.49 % | 22.10 % | 5.09 Tbyte/s | 256 | 31 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp16 | cutile | 19.42 us | 57.74 % | 57.74 % | 24.64 % | 37.94 % | 40.38 % | 4.42 Tbyte/s | 128 | 74 register/thread | 28 byte/block | 0 byte/block | 6 block / 14 block |
| bf16 | triton | 16.42 us | 68.43 % | 68.43 % | 24.04 % | 44.67 % | 25.74 % | 5.23 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| bf16 | cutile | 26.78 us | 41.86 % | 41.86 % | 14.50 % | 27.33 % | 40.24 % | 3.20 Tbyte/s | 128 | 50 register/thread | 28 byte/block | 0 byte/block | 9 block / 28 block |
| fp32 | triton | 28.70 us | 77.85 % | 77.85 % | 22.67 % | 50.78 % | 7.82 % | 5.97 Tbyte/s | 256 | 28 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp32 | cutile | 32.54 us | 68.62 % | 68.62 % | 19.91 % | 45.01 % | 30.33 % | 5.26 Tbyte/s | 128 | 78 register/thread | 28 byte/block | 0 byte/block | 6 block / 14 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.16× faster** (16.8 µs vs 19.4 µs).
- **bf16**: Triton is **1.63× faster** (16.4 µs vs 26.8 µs).
- **fp32**: Triton is **1.13× faster** (28.7 µs vs 32.5 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — Memory is more heavily utilized than Compute
- **fp32 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
