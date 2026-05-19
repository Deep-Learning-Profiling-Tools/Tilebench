# NCU Comparison: matmul_int8

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| int8 | `{'M': 2048, 'N': 2048, 'K': 20480}` | `{'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}` | `{'tm': 256, 'tn': 64, 'tk': 32, 'group_size_m': 8, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| int8 | triton | 475.49 us | 50.98 % | 1.50 % | 56.68 % | 12.69 % | 40.42 % | 115.29 Gbyte/s | 128 | 255 register/thread | 0 byte/block | 36.86 Kbyte/block | 2 block / 3 block |
| int8 | cutile | 342.37 us | 55.01 % | 2.19 % | 65.13 % | 34.00 % | 29.98 % | 168.00 Gbyte/s | 256 | 255 register/thread | 141.71 Kbyte/block | 0 byte/block | 1 block / 1 block |

## Key findings (auto-derived)

- **int8**: cuTile is **1.39× faster** (342.4 µs vs 475.5 µs).

## NCU's own bottleneck verdict

- **int8 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **int8 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.86 full waves across all SMs. Look at Launch Statistics for more details.

## Reports

- `cutile_int8.ncu-rep`
- `triton_int8.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
