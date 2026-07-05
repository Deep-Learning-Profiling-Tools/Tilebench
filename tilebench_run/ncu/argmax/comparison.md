# NCU Comparison: argmax

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_N': 2048, 'num_warps': 4, 'num_stages': 4}` | `{'block_n': 2048, 'occupancy': 16}` |
| fp32 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_N': 1024, 'num_warps': 4, 'num_stages': 4}` | `{'block_n': 2048, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 31.20 us | 35.99 % | 35.99 % | 44.26 % | 23.44 % | 57.72 % | 2.75 Tbyte/s | 128 | 38 register/thread | 0 byte/block | 32 byte/block | 12 block / 28 block |
| fp16 | cutile | 30.91 us | 36.23 % | 36.23 % | 42.39 % | 23.62 % | 54.82 % | 2.78 Tbyte/s | 128 | 31 register/thread | 44 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 41.09 us | 54.23 % | 54.23 % | 59.08 % | 35.54 % | 59.52 % | 4.16 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 32 byte/block | 16 block / 28 block |
| fp32 | cutile | 30.78 us | 72.46 % | 72.46 % | 53.57 % | 47.57 % | 51.54 % | 5.55 Tbyte/s | 128 | 32 register/thread | 44 byte/block | 0 byte/block | 16 block / 28 block |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.01× faster** (30.9 µs vs 31.2 µs).
- **fp32**: cuTile is **1.33× faster** (30.8 µs vs 41.1 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — Memory is more heavily utilized than Compute
- **fp32 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
