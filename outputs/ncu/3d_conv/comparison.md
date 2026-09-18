# NCU Comparison: 3d_conv

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'in_channels': 64, 'out_channels': 64, 'D': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1, 'H': 320}` | `{'BLOCK_SIZE_BATCH_DHW': 128, 'BLOCK_SIZE_IN_FEAT': 64, 'BLOCK_SIZE_OUT_FEAT': 64, 'num_warps': 4, 'num_stages': 2}` | `{'block_bdhw': 128, 'block_in': 16, 'block_out': 64, 'occupancy': 4}` |
| fp32 | `{'batch': 1, 'in_channels': 64, 'out_channels': 64, 'D': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1, 'H': 320}` | `{'BLOCK_SIZE_BATCH_DHW': 64, 'BLOCK_SIZE_IN_FEAT': 64, 'BLOCK_SIZE_OUT_FEAT': 64, 'num_warps': 4, 'num_stages': 2}` | `{'block_bdhw': 128, 'block_in': 16, 'block_out': 64, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 13850.00 us | 18.92 % | 0.89 % | 18.98 % | 3.11 % | 67.56 % | 68.29 Gbyte/s | 128 | 255 register/thread | 0 byte/block | 49.17 Kbyte/block | 2 block / 4 block |
| fp16 | cutile | 35670.00 us | 5.54 % | 0.36 % | 5.56 % | 2.11 % | 71.32 % | 27.52 Gbyte/s | 256 | 64 register/thread | 2.09 Kbyte/block | 0 byte/block | 4 block / 10 block |
| fp32 | triton | 18200.00 us | 55.12 % | 2.23 % | 55.23 % | 7.57 % | 63.80 % | 171.24 Gbyte/s | 128 | 165 register/thread | 0 byte/block | 65.55 Kbyte/block | 3 block / 3 block |
| fp32 | cutile | 33080.00 us | 5.89 % | 1.24 % | 5.90 % | 2.63 % | 76.31 % | 94.80 Gbyte/s | 256 | 64 register/thread | 4.14 Kbyte/block | 0 byte/block | 4 block / 12 block |

## Key findings (auto-derived)

- **fp16**: Triton is **2.58× faster** (13850.0 µs vs 35670.0 µs).
- **fp32**: Triton is **1.82× faster** (18200.0 µs vs 33080.0 µs).

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
