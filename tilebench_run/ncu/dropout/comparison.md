# NCU Comparison: dropout

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'p': 0.5, 'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4}` | `{'tile': 2048, 'occupancy': 16}` |
| bf16 | `{'p': 0.5, 'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'tile': 2048, 'occupancy': 16}` |
| fp32 | `{'p': 0.5, 'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 8}` | `{'tile': 2048, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 18.37 us | 67.64 % | 67.64 % | 40.86 % | 40.03 % | 28.55 % | 5.18 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 18.78 us | 66.86 % | 66.86 % | 40.02 % | 39.08 % | 32.68 % | 5.11 Tbyte/s | 128 | 26 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | triton | 18.46 us | 68.41 % | 68.41 % | 40.68 % | 39.70 % | 35.40 % | 5.23 Tbyte/s | 256 | 19 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| bf16 | cutile | 18.85 us | 66.66 % | 66.66 % | 39.82 % | 38.93 % | 35.31 % | 5.11 Tbyte/s | 128 | 27 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | triton | 34.59 us | 80.75 % | 80.75 % | 38.28 % | 42.60 % | 21.46 % | 6.19 Tbyte/s | 256 | 20 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 36.10 us | 77.72 % | 77.72 % | 36.28 % | 40.87 % | 19.93 % | 5.96 Tbyte/s | 128 | 24 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.02× faster** (18.4 µs vs 18.8 µs).
- **bf16**: Triton is **1.02× faster** (18.5 µs vs 18.9 µs).
- **fp32**: Triton is **1.04× faster** (34.6 µs vs 36.1 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Memory is more heavily utilized than Compute
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Memory is more heavily utilized than Compute
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — Memory is more heavily utilized than Compute
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
