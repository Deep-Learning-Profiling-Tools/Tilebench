# NCU Comparison: 2d_conv

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1, 'H': 320}` | `{'BLOCK_SIZE_BATCH_HEIGHT_WIDTH': 128, 'BLOCK_SIZE_IN_FEAT': 64, 'BLOCK_SIZE_OUT_FEAT': 64, 'num_warps': 4, 'num_stages': 3}` | `{'block_bhw': 32, 'block_in': 32, 'block_out': 128, 'occupancy': 4}` |
| fp32 | `{'batch': 1, 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1, 'H': 320}` | `{'BLOCK_SIZE_BATCH_HEIGHT_WIDTH': 64, 'BLOCK_SIZE_IN_FEAT': 16, 'BLOCK_SIZE_OUT_FEAT': 128, 'num_warps': 4, 'num_stages': 2}` | `{'block_bhw': 32, 'block_in': 32, 'block_out': 128, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 556.13 us | 20.98 % | 0.63 % | 21.79 % | 2.95 % | 64.99 % | 48.07 Gbyte/s | 128 | 253 register/thread | 0 byte/block | 49.17 Kbyte/block | 2 block / 4 block |
| fp16 | cutile | 797.63 us | 20.85 % | 0.44 % | 21.29 % | 5.26 % | 72.50 % | 33.63 Gbyte/s | 128 | 114 register/thread | 18.44 Kbyte/block | 0 byte/block | 4 block / 6 block |
| fp32 | triton | 1860.00 us | 59.33 % | 0.48 % | 60.78 % | 2.44 % | 39.59 % | 36.55 Gbyte/s | 128 | 205 register/thread | 0 byte/block | 32.77 Kbyte/block | 2 block / 4 block |
| fp32 | cutile | 5390.00 us | 50.45 % | 41.44 % | 51.80 % | 41.22 % | 34.09 % | 3.18 Tbyte/s | 128 | 128 register/thread | 34.83 Kbyte/block | 0 byte/block | 4 block / 4 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.43× faster** (556.1 µs vs 797.6 µs).
- **fp32**: Triton is **2.90× faster** (1860.0 µs vs 5390.0 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — Compute is more heavily utilized than Memory
- **fp16 / triton** — Compute is more heavily utilized than Memory
- **fp32 / cutile** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent
- **fp32 / triton** — This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potent

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
