# NCU Comparison: dropout

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'p': 0.5, 'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'tile': 2048, 'occupancy': 16}` |
| bf16 | `{'p': 0.5, 'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'tile': 2048, 'occupancy': 16}` |
| fp32 | `{'p': 0.5, 'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 2}` | `{'tile': 512, 'occupancy': 32}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 18.27 us | 68.22 % | 68.22 % | 40.71 % | 40.30 % | 30.96 % | 5.23 Tbyte/s | 64 | 32 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| fp16 | cutile | 18.56 us | 67.55 % | 67.55 % | 39.47 % | 39.61 % | 32.35 % | 5.17 Tbyte/s | 128 | 26 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | triton | 18.05 us | 69.11 % | 69.11 % | 40.99 % | 40.87 % | 32.00 % | 5.29 Tbyte/s | 64 | 32 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |
| bf16 | cutile | 19.07 us | 65.69 % | 65.69 % | 39.73 % | 38.49 % | 35.37 % | 5.03 Tbyte/s | 128 | 27 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | triton | 34.94 us | 79.34 % | 79.34 % | 36.92 % | 41.84 % | 13.70 % | 6.08 Tbyte/s | 64 | 34 register/thread | 0 byte/block | 0 byte/block | 24 block / 32 block |
| fp32 | cutile | 34.53 us | 80.95 % | 80.95 % | 37.50 % | 42.52 % | 38.05 % | 6.21 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.02× faster** (18.3 µs vs 18.6 µs).
- **bf16**: Triton is **1.06× faster** (18.1 µs vs 19.1 µs).
- **fp32**: cuTile is **1.01× faster** (34.5 µs vs 34.9 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Memory is more heavily utilized than Compute
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Memory is more heavily utilized than Compute
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
- **fp32 / triton** — Memory is more heavily utilized than Compute

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
