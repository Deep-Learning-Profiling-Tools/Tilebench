# NCU Comparison: 1d_conv

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'batch': 1, 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1, 'L': 2621440}` | `{'BLOCK_SIZE_BATCH_LENGTH': 128, 'BLOCK_SIZE_IN_FEAT': 32, 'BLOCK_SIZE_OUT_FEAT': 128, 'num_warps': 8, 'num_stages': 3}` | `{'block_bl': 32, 'block_in': 32, 'block_out': 128, 'occupancy': 8}` |
| fp32 | `{'batch': 1, 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 1, 'L': 2621440}` | `{'BLOCK_SIZE_BATCH_LENGTH': 64, 'BLOCK_SIZE_IN_FEAT': 16, 'BLOCK_SIZE_OUT_FEAT': 128, 'num_warps': 4, 'num_stages': 3}` | `{'block_bl': 128, 'block_in': 32, 'block_out': 64, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 2520.00 us | 22.92 % | 6.73 % | 23.04 % | 6.19 % | 64.00 % | 516.17 Gbyte/s | 256 | 128 register/thread | 0 byte/block | 32.78 Kbyte/block | 2 block / 3 block |
| fp16 | cutile | 4510.00 us | 48.23 % | 3.84 % | 48.36 % | 25.81 % | 77.41 % | 294.64 Gbyte/s | 128 | 64 register/thread | 18.44 Kbyte/block | 0 byte/block | 8 block / 8 block |
| fp32 | triton | 3080.00 us | 44.10 % | 11.17 % | 44.23 % | 16.02 % | 66.39 % | 857.01 Gbyte/s | 128 | 124 register/thread | 0 byte/block | 36.88 Kbyte/block | 4 block / 4 block |
| fp32 | cutile | 5320.00 us | 11.06 % | 9.80 % | 11.10 % | 6.36 % | 77.92 % | 751.63 Gbyte/s | 256 | 64 register/thread | 8.24 Kbyte/block | 0 byte/block | 4 block / 10 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.79× faster** (2520.0 µs vs 4510.0 µs).
- **fp32**: Triton is **1.73× faster** (3080.0 µs vs 5320.0 µs).

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
