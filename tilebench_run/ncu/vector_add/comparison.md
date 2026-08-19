# NCU Comparison: vector_add

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'tile': 1024, 'occupancy': 16}` |
| bf16 | `{'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'tile': 2048, 'occupancy': 16}` |
| fp32 | `{'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 8}` | `{'tile': 512, 'occupancy': 16}` |
| int8 | `{'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4}` | `{'tile': 2048, 'occupancy': 32}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 24.03 us | 73.32 % | 73.32 % | 29.10 % | 33.19 % | 11.89 % | 5.62 Tbyte/s | 256 | 20 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp16 | cutile | 23.94 us | 71.52 % | 71.52 % | 27.04 % | 33.04 % | 25.72 % | 5.47 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | triton | 23.74 us | 71.08 % | 71.08 % | 29.47 % | 33.06 % | 6.07 % | 5.44 Tbyte/s | 64 | 26 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | cutile | 24.70 us | 68.33 % | 68.33 % | 28.80 % | 31.88 % | 20.03 % | 5.23 Tbyte/s | 128 | 28 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | triton | 41.54 us | 81.99 % | 81.99 % | 30.32 % | 37.73 % | 13.87 % | 6.28 Tbyte/s | 256 | 20 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 41.63 us | 81.68 % | 81.68 % | 30.15 % | 37.74 % | 29.66 % | 6.26 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| int8 | triton | 14.75 us | 59.47 % | 59.47 % | 26.25 % | 27.19 % | 33.45 % | 4.55 Tbyte/s | 128 | 28 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| int8 | cutile | 15.97 us | 55.49 % | 55.49 % | 23.16 % | 25.05 % | 37.73 % | 4.24 Tbyte/s | 128 | 23 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.00× faster** (23.9 µs vs 24.0 µs).
- **bf16**: Triton is **1.04× faster** (23.7 µs vs 24.7 µs).
- **fp32**: Triton is **1.00× faster** (41.5 µs vs 41.6 µs).
- **int8**: Triton is **1.08× faster** (14.8 µs vs 16.0 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Memory is more heavily utilized than Compute
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Memory is more heavily utilized than Compute
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
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
