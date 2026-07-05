# NCU Comparison: l2_norm

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'M': 2048, 'eps': 1e-06, 'K': 10240}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 3}` | `{'tile_size': 1024, 'occupancy': 32}` |
| bf16 | `{'batch': 1, 'M': 2048, 'eps': 1e-06, 'K': 10240}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 4}` | `{'tile_size': 1024, 'occupancy': 32}` |
| fp32 | `{'batch': 1, 'M': 2048, 'eps': 1e-06, 'K': 10240}` | `{'BLOCK_N': 2048, 'num_warps': 8, 'num_stages': 4}` | `{'tile_size': 1024, 'occupancy': 32}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 14.21 us | 40.65 % | 40.65 % | 56.62 % | 35.57 % | 25.75 % | 3.11 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp16 | cutile | 18.62 us | 31.34 % | 31.34 % | 40.91 % | 28.72 % | 37.32 % | 2.40 Tbyte/s | 128 | 24 register/thread | 28 byte/block | 0 byte/block | 21 block / 28 block |
| bf16 | triton | 14.50 us | 40.24 % | 40.24 % | 56.97 % | 34.82 % | 27.24 % | 3.07 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| bf16 | cutile | 19.39 us | 30.30 % | 30.30 % | 40.55 % | 27.75 % | 37.26 % | 2.32 Tbyte/s | 128 | 24 register/thread | 28 byte/block | 0 byte/block | 21 block / 28 block |
| fp32 | triton | 27.46 us | 57.81 % | 57.81 % | 50.94 % | 38.53 % | 10.40 % | 4.43 Tbyte/s | 256 | 36 register/thread | 0 byte/block | 32 byte/block | 6 block / 14 block |
| fp32 | cutile | 42.50 us | 60.25 % | 60.25 % | 50.88 % | 39.91 % | 14.65 % | 4.61 Tbyte/s | 128 | 24 register/thread | 28 byte/block | 0 byte/block | 21 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.31× faster** (14.2 µs vs 18.6 µs).
- **bf16**: Triton is **1.34× faster** (14.5 µs vs 19.4 µs).
- **fp32**: Triton is **1.55× faster** (27.5 µs vs 42.5 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **bf16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp16 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
- **fp16 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / cutile** — Memory is more heavily utilized than Compute
- **fp32 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
