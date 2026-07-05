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
| fp16 | triton | 18.75 us | 66.21 % | 66.21 % | 40.38 % | 39.26 % | 30.60 % | 5.06 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 18.85 us | 66.57 % | 66.57 % | 39.93 % | 38.94 % | 32.41 % | 5.10 Tbyte/s | 128 | 26 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | triton | 18.43 us | 68.52 % | 68.52 % | 40.98 % | 39.81 % | 36.88 % | 5.25 Tbyte/s | 256 | 19 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| bf16 | cutile | 19.01 us | 66.12 % | 66.12 % | 39.74 % | 38.66 % | 34.26 % | 5.06 Tbyte/s | 128 | 27 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | triton | 34.21 us | 81.48 % | 81.48 % | 38.23 % | 43.04 % | 21.08 % | 6.24 Tbyte/s | 256 | 20 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 35.74 us | 78.67 % | 78.67 % | 36.05 % | 41.26 % | 19.35 % | 6.03 Tbyte/s | 128 | 24 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.01× faster** (18.8 µs vs 18.9 µs).
- **bf16**: Triton is **1.03× faster** (18.4 µs vs 19.0 µs).
- **fp32**: Triton is **1.04× faster** (34.2 µs vs 35.7 µs).

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
