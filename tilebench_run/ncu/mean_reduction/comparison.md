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
| fp16 | triton | 16.86 us | 66.38 % | 66.38 % | 21.60 % | 43.52 % | 20.36 % | 5.08 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp16 | cutile | 58.78 us | 19.04 % | 19.04 % | 5.86 % | 12.36 % | 11.16 % | 1.46 Tbyte/s | 128 | 142 register/thread | 28 byte/block | 0 byte/block | 3 block / 7 block |
| bf16 | triton | 19.04 us | 59.04 % | 59.04 % | 19.05 % | 38.57 % | 23.47 % | 4.52 Tbyte/s | 128 | 25 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| bf16 | cutile | 20.74 us | 54.02 % | 54.02 % | 18.29 % | 35.39 % | 35.37 % | 4.14 Tbyte/s | 128 | 58 register/thread | 28 byte/block | 0 byte/block | 8 block / 28 block |
| fp32 | triton | 29.98 us | 74.52 % | 74.52 % | 20.27 % | 48.75 % | 10.92 % | 5.71 Tbyte/s | 128 | 23 register/thread | 0 byte/block | 16 byte/block | 21 block / 28 block |
| fp32 | cutile | 30.05 us | 74.33 % | 74.33 % | 20.39 % | 48.71 % | 22.29 % | 5.69 Tbyte/s | 128 | 30 register/thread | 28 byte/block | 0 byte/block | 16 block / 28 block |

## Key findings (auto-derived)

- **fp16**: Triton is **3.49× faster** (16.9 µs vs 58.8 µs).
- **bf16**: Triton is **1.09× faster** (19.0 µs vs 20.7 µs).
- **fp32**: Triton is **1.00× faster** (30.0 µs vs 30.1 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **bf16 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.
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
