# NCU Comparison: rope

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch_size': 1, 'n_heads': 32, 'head_dim': 128, 'seq_len': 20480}` | `{'ROPE_GROUP_SIZE': 16, 'num_warps': 4, 'num_stages': 3}` | `{'group_size': 16, 'occupancy': 8}` |
| fp32 | `{'batch_size': 1, 'n_heads': 32, 'head_dim': 128, 'seq_len': 20480}` | `{'ROPE_GROUP_SIZE': 16, 'num_warps': 4, 'num_stages': 2}` | `{'group_size': 16, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 54.46 us | 77.94 % | 77.94 % | 42.46 % | 37.10 % | 21.94 % | 5.98 Tbyte/s | 128 | 27 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 55.14 us | 77.05 % | 77.05 % | 46.86 % | 36.72 % | 65.05 % | 5.90 Tbyte/s | 128 | 32 register/thread | 1.16 Kbyte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 106.66 us | 82.90 % | 82.90 % | 40.90 % | 38.51 % | 15.33 % | 6.36 Tbyte/s | 128 | 40 register/thread | 0 byte/block | 0 byte/block | 12 block / 32 block |
| fp32 | cutile | 106.43 us | 83.07 % | 83.07 % | 41.33 % | 38.58 % | 36.76 % | 6.37 Tbyte/s | 128 | 46 register/thread | 1.29 Kbyte/block | 0 byte/block | 10 block / 26 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.01× faster** (54.5 µs vs 55.1 µs).
- **fp32**: cuTile is **1.00× faster** (106.4 µs vs 106.7 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — Memory is more heavily utilized than Compute
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
