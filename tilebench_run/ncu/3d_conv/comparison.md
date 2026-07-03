# NCU Comparison: 3d_conv

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'in_channels': 64, 'out_channels': 64, 'D': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1, 'H': 320}` | `{'BLOCK_SIZE_BATCH_DHW': 128, 'BLOCK_SIZE_IN_FEAT': 64, 'BLOCK_SIZE_OUT_FEAT': 64, 'num_warps': 4, 'num_stages': 3}` | `{'block_bdhw': 128, 'block_in': 16, 'block_out': 64, 'occupancy': 4}` |
| fp32 | `{'batch': 1, 'in_channels': 64, 'out_channels': 64, 'D': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1, 'H': 320}` | `{'BLOCK_SIZE_BATCH_DHW': 64, 'BLOCK_SIZE_IN_FEAT': 64, 'BLOCK_SIZE_OUT_FEAT': 64, 'num_warps': 4, 'num_stages': 2}` | `{'block_bdhw': 128, 'block_in': 16, 'block_out': 64, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 13870.00 us | 18.91 % | 0.90 % | 18.96 % | 3.09 % | 67.53 % | 68.96 Gbyte/s | 128 | 255 register/thread | 0 byte/block | 49.17 Kbyte/block | 2 block / 4 block |
| fp16 | cutile | 35770.00 us | 5.53 % | 0.36 % | 5.54 % | 2.10 % | 71.13 % | 27.51 Gbyte/s | 256 | 64 register/thread | 2.09 Kbyte/block | 0 byte/block | 4 block / 10 block |
| fp32 | triton | 18190.00 us | 55.16 % | 2.26 % | 55.23 % | 7.57 % | 63.79 % | 173.02 Gbyte/s | 128 | 165 register/thread | 0 byte/block | 65.55 Kbyte/block | 3 block / 3 block |
| fp32 | cutile | 33160.00 us | 5.87 % | 1.24 % | 5.89 % | 2.63 % | 76.15 % | 95.35 Gbyte/s | 256 | 64 register/thread | 4.14 Kbyte/block | 0 byte/block | 4 block / 12 block |

## Key findings (auto-derived)

- **fp16**: Triton is **2.58× faster** (13870.0 µs vs 35770.0 µs).
- **fp32**: Triton is **1.82× faster** (18190.0 µs vs 33160.0 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — Compute is more heavily utilized than Memory
- **fp16 / triton** — Compute is more heavily utilized than Memory
- **fp32 / cutile** — Compute is more heavily utilized than Memory
- **fp32 / triton** — Compute and Memory are well-balanced

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
