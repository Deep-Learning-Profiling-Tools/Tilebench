# NCU Comparison: argmax

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_N': 2048, 'num_warps': 4, 'num_stages': 4}` | `{'block_n': 2048, 'occupancy': 16}` |
| fp32 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_N': 1024, 'num_warps': 4, 'num_stages': 2}` | `{'block_n': 2048, 'occupancy': 32}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 32.48 us | 49.25 % | 49.25 % | 39.27 % | 22.47 % | 53.93 % | 3.77 Tbyte/s | 128 | 38 register/thread | 0 byte/block | 32 byte/block | 12 block / 28 block |
| fp16 | cutile | 33.18 us | 48.41 % | 48.41 % | 39.22 % | 22.01 % | 52.37 % | 3.71 Tbyte/s | 128 | 31 register/thread | 44 byte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 43.97 us | 67.21 % | 67.21 % | 54.40 % | 33.22 % | 55.76 % | 5.15 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 32 byte/block | 16 block / 28 block |
| fp32 | cutile | 38.82 us | 74.81 % | 74.81 % | 43.31 % | 37.59 % | 41.60 % | 5.74 Tbyte/s | 128 | 24 register/thread | 44 byte/block | 0 byte/block | 21 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.02× faster** (32.5 µs vs 33.2 µs).
- **fp32**: cuTile is **1.13× faster** (38.8 µs vs 44.0 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — Memory is more heavily utilized than Compute
- **fp32 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
