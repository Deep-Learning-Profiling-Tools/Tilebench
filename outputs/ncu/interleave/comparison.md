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
| fp16 | triton | 28.45 us | 74.05 % | 74.05 % | 43.59 % | 33.93 % | 15.54 % | 5.67 Tbyte/s | 128 | 25 register/thread | 0 byte/block | 4.10 Kbyte/block | 16 block / 26 block |
| fp16 | cutile | 29.50 us | 70.74 % | 70.74 % | 44.21 % | 32.51 % | 19.65 % | 5.42 Tbyte/s | 128 | 62 register/thread | 16.40 Kbyte/block | 0 byte/block | 8 block / 9 block |
| bf16 | triton | 28.54 us | 73.68 % | 73.68 % | 43.17 % | 33.60 % | 15.47 % | 5.65 Tbyte/s | 128 | 25 register/thread | 0 byte/block | 4.10 Kbyte/block | 16 block / 26 block |
| bf16 | cutile | 29.09 us | 72.15 % | 72.15 % | 45.64 % | 33.04 % | 31.86 % | 5.53 Tbyte/s | 128 | 25 register/thread | 4.11 Kbyte/block | 0 byte/block | 16 block / 25 block |
| fp32 | triton | 51.74 us | 81.01 % | 81.01 % | 44.71 % | 36.99 % | 16.94 % | 6.21 Tbyte/s | 256 | 22 register/thread | 0 byte/block | 8.19 Kbyte/block | 10 block / 14 block |
| fp32 | cutile | 51.84 us | 80.49 % | 80.49 % | 45.66 % | 36.94 % | 17.01 % | 6.17 Tbyte/s | 128 | 52 register/thread | 16.40 Kbyte/block | 0 byte/block | 9 block / 9 block |
| int8 | triton | 16.96 us | 61.63 % | 61.63 % | 44.41 % | 28.36 % | 12.96 % | 4.72 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 4.10 Kbyte/block | 16 block / 26 block |
| int8 | cutile | 17.41 us | 59.74 % | 59.74 % | 41.57 % | 27.69 % | 25.47 % | 4.57 Tbyte/s | 128 | 23 register/thread | 4.11 Kbyte/block | 0 byte/block | 21 block / 25 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.04× faster** (28.4 µs vs 29.5 µs).
- **bf16**: Triton is **1.02× faster** (28.5 µs vs 29.1 µs).
- **fp32**: Triton is **1.00× faster** (51.7 µs vs 51.8 µs).
- **int8**: Triton is **1.03× faster** (17.0 µs vs 17.4 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Memory is more heavily utilized than Compute
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Memory is more heavily utilized than Compute
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **int8 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **int8 / triton** — Memory is more heavily utilized than Compute

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
