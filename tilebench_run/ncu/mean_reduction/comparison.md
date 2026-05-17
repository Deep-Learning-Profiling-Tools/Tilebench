# NCU Comparison: mean_reduction

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_M': 1, 'BLOCK_N': 512, 'num_warps': 4}` | `{'tile_size': 512, 'occupancy': 4}` |
| bf16 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_M': 1, 'BLOCK_N': 512, 'num_warps': 4}` | `{'tile_size': 512, 'occupancy': 4}` |
| fp32 | `{'M': 2048, 'N': 20480}` | `{'BLOCK_M': 1, 'BLOCK_N': 1024, 'num_warps': 8}` | `{'tile_size': 512, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 17.12 us | 65.49 % | 65.49 % | 21.15 % | 42.92 % | 28.57 % | 5.01 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| fp16 | cutile | 19.62 us | 57.13 % | 57.13 % | 24.60 % | 37.56 % | 40.56 % | 4.38 Tbyte/s | 128 | 74 register/thread | 28 byte/block | 0 byte/block | 6 block / 14 block |
| bf16 | triton | 17.18 us | 65.29 % | 65.29 % | 21.43 % | 42.84 % | 32.78 % | 4.99 Tbyte/s | 128 | 31 register/thread | 0 byte/block | 16 byte/block | 16 block / 28 block |
| bf16 | cutile | 26.62 us | 42.08 % | 42.08 % | 14.45 % | 27.50 % | 39.13 % | 3.22 Tbyte/s | 128 | 50 register/thread | 28 byte/block | 0 byte/block | 9 block / 28 block |
| fp32 | triton | 29.86 us | 74.73 % | 74.73 % | 22.12 % | 48.94 % | 14.59 % | 5.73 Tbyte/s | 256 | 32 register/thread | 0 byte/block | 32 byte/block | 8 block / 28 block |
| fp32 | cutile | 32.67 us | 68.38 % | 68.38 % | 19.81 % | 44.73 % | 30.82 % | 5.23 Tbyte/s | 128 | 78 register/thread | 28 byte/block | 0 byte/block | 6 block / 14 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.15× faster** (17.1 µs vs 19.6 µs).
- **bf16**: Triton is **1.55× faster** (17.2 µs vs 26.6 µs).
- **fp32**: Triton is **1.09× faster** (29.9 µs vs 32.7 µs).

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
