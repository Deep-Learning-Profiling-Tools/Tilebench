# NCU Comparison: flash_decode

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'batch': 2, 'heads': 8, 'head_dim': 128, 'block_seq': 128, 'seq_len': 40960}` | `{'num_warps': 4, 'num_stages': 2}` | `{'occupancy': 32}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 47.42 us | 0.73 % | 0.73 % | 3.04 % | 0.48 % | 0.89 % | 55.85 Gbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | cutile | 168.86 us | 0.20 % | 0.20 % | 0.81 % | 0.14 % | 0.52 % | 15.69 Gbyte/s | 128 | 24 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |

## Key findings (auto-derived)

- **fp32**: Triton is **3.56× faster** (47.4 µs vs 168.9 µs).

## NCU's own bottleneck verdict

- **fp32 / cutile** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.01 full waves across all SMs. Look at Launch Statistics for more details.
- **fp32 / triton** — This kernel grid is too small to fill the available resources on this device, resulting in only 0.01 full waves across all SMs. Look at Launch Statistics for more details.

## Reports

- `cutile_fp32.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
