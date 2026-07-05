# NCU Comparison: vector_add

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'tile': 1024, 'occupancy': 32}` |
| bf16 | `{'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 4}` | `{'tile': 1024, 'occupancy': 32}` |
| fp32 | `{'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 8}` | `{'tile': 1024, 'occupancy': 32}` |
| int8 | `{'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4}` | `{'tile': 1024, 'occupancy': 32}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 18.27 us | 68.92 % | 68.92 % | 40.94 % | 40.10 % | 15.40 % | 5.28 Tbyte/s | 256 | 20 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp16 | cutile | 18.43 us | 68.37 % | 68.37 % | 40.31 % | 39.91 % | 35.20 % | 5.24 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | triton | 18.05 us | 69.59 % | 69.59 % | 41.16 % | 40.77 % | 16.05 % | 5.33 Tbyte/s | 128 | 20 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| bf16 | cutile | 18.62 us | 67.59 % | 67.59 % | 40.31 % | 39.42 % | 35.50 % | 5.18 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| fp32 | triton | 34.34 us | 81.29 % | 81.29 % | 37.83 % | 43.02 % | 16.57 % | 6.23 Tbyte/s | 256 | 20 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 34.40 us | 80.79 % | 80.79 % | 38.01 % | 42.73 % | 23.81 % | 6.19 Tbyte/s | 128 | 20 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| int8 | triton | 12.03 us | 45.70 % | 45.70 % | 35.30 % | 30.68 % | 43.68 % | 3.50 Tbyte/s | 128 | 28 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| int8 | cutile | 17.18 us | 32.03 % | 32.03 % | 22.95 % | 21.27 % | 53.55 % | 2.45 Tbyte/s | 128 | 22 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.01× faster** (18.3 µs vs 18.4 µs).
- **bf16**: Triton is **1.03× faster** (18.1 µs vs 18.6 µs).
- **fp32**: Triton is **1.00× faster** (34.3 µs vs 34.4 µs).
- **int8**: Triton is **1.43× faster** (12.0 µs vs 17.2 µs).

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
