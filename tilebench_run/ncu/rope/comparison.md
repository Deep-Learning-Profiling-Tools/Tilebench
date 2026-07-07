# NCU Comparison: rope

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch_size': 1, 'n_heads': 32, 'head_dim': 128, 'seq_len': 20480}` | `{'ROPE_GROUP_SIZE': 16, 'num_warps': 4, 'num_stages': 2}` | `{'group_size': 16, 'occupancy': 16}` |
| fp32 | `{'batch_size': 1, 'n_heads': 32, 'head_dim': 128, 'seq_len': 20480}` | `{'ROPE_GROUP_SIZE': 16, 'num_warps': 4, 'num_stages': 2}` | `{'group_size': 16, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 49.38 us | 75.92 % | 75.92 % | 47.41 % | 39.47 % | 24.18 % | 5.82 Tbyte/s | 128 | 27 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 52.48 us | 71.26 % | 71.26 % | 50.00 % | 36.99 % | 72.28 % | 5.46 Tbyte/s | 128 | 28 register/thread | 1.16 Kbyte/block | 0 byte/block | 16 block / 28 block |
| fp32 | triton | 101.66 us | 81.45 % | 81.45 % | 43.00 % | 39.54 % | 16.10 % | 6.25 Tbyte/s | 128 | 40 register/thread | 0 byte/block | 0 byte/block | 12 block / 32 block |
| fp32 | cutile | 106.27 us | 79.19 % | 79.19 % | 50.94 % | 41.09 % | 40.57 % | 6.07 Tbyte/s | 128 | 32 register/thread | 1.29 Kbyte/block | 0 byte/block | 16 block / 42 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.06× faster** (49.4 µs vs 52.5 µs).
- **fp32**: Triton is **1.05× faster** (101.7 µs vs 106.3 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — Compute and Memory are well-balanced
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — Memory is more heavily utilized than Compute
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
