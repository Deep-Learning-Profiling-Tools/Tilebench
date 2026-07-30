# NCU Comparison: leaky_relu

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 50000000}` | `{'BLOCK_SIZE': 4096, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 8192, 'occupancy': 4}` |
| bf16 | `{'n': 50000000}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 8192, 'occupancy': 4}` |
| fp32 | `{'n': 50000000}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8, 'num_stages': 2}` | `{'tile': 8192, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 26.34 us | 70.93 % | 70.93 % | 55.11 % | 40.98 % | 27.91 % | 5.43 Tbyte/s | 128 | 38 register/thread | 0 byte/block | 0 byte/block | 12 block / 32 block |
| fp16 | cutile | 27.23 us | 68.84 % | 68.84 % | 55.07 % | 39.48 % | 48.13 % | 5.27 Tbyte/s | 128 | 56 register/thread | 0 byte/block | 0 byte/block | 9 block / 32 block |
| bf16 | triton | 27.10 us | 69.20 % | 69.20 % | 56.03 % | 39.93 % | 27.40 % | 5.30 Tbyte/s | 128 | 24 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| bf16 | cutile | 28.00 us | 66.94 % | 66.94 % | 55.65 % | 38.46 % | 47.81 % | 5.13 Tbyte/s | 128 | 56 register/thread | 0 byte/block | 0 byte/block | 9 block / 32 block |
| fp32 | triton | 55.78 us | 81.33 % | 81.33 % | 47.76 % | 40.86 % | 12.34 % | 6.23 Tbyte/s | 256 | 18 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 58.43 us | 77.15 % | 77.15 % | 47.02 % | 38.87 % | 24.45 % | 5.92 Tbyte/s | 128 | 82 register/thread | 0 byte/block | 0 byte/block | 5 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.03× faster** (26.3 µs vs 27.2 µs).
- **bf16**: Triton is **1.03× faster** (27.1 µs vs 28.0 µs).
- **fp32**: Triton is **1.05× faster** (55.8 µs vs 58.4 µs).

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
