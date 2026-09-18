# NCU Comparison: 2d_conv

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1, 'H': 320}` | `{'BLOCK_SIZE_BATCH_HEIGHT_WIDTH': 128, 'BLOCK_SIZE_IN_FEAT': 64, 'BLOCK_SIZE_OUT_FEAT': 64, 'num_warps': 4, 'num_stages': 4}` | `{'block_bhw': 32, 'block_in': 32, 'block_out': 128, 'occupancy': 4}` |
| fp32 | `{'batch': 1, 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1, 'H': 320}` | `{'BLOCK_SIZE_BATCH_HEIGHT_WIDTH': 64, 'BLOCK_SIZE_IN_FEAT': 16, 'BLOCK_SIZE_OUT_FEAT': 128, 'num_warps': 4, 'num_stages': 3}` | `{'block_bhw': 32, 'block_in': 32, 'block_out': 128, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 552.00 us | 20.96 % | 0.64 % | 21.78 % | 2.98 % | 64.94 % | 49.22 Gbyte/s | 128 | 253 register/thread | 0 byte/block | 49.17 Kbyte/block | 2 block / 4 block |
| fp16 | cutile | 797.22 us | 20.82 % | 0.44 % | 21.30 % | 5.22 % | 72.40 % | 33.68 Gbyte/s | 128 | 114 register/thread | 18.44 Kbyte/block | 0 byte/block | 4 block / 6 block |
| fp32 | triton | 626.53 us | 22.39 % | 1.35 % | 23.02 % | 6.86 % | 60.96 % | 103.85 Gbyte/s | 128 | 124 register/thread | 0 byte/block | 36.88 Kbyte/block | 4 block / 4 block |
| fp32 | cutile | 934.14 us | 27.82 % | 0.97 % | 28.49 % | 7.64 % | 66.11 % | 74.47 Gbyte/s | 128 | 128 register/thread | 34.83 Kbyte/block | 0 byte/block | 4 block / 4 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.44× faster** (552.0 µs vs 797.2 µs).
- **fp32**: Triton is **1.49× faster** (626.5 µs vs 934.1 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — Compute is more heavily utilized than Memory
- **fp16 / triton** — Compute is more heavily utilized than Memory
- **fp32 / cutile** — Compute is more heavily utilized than Memory
- **fp32 / triton** — Compute is more heavily utilized than Memory

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
