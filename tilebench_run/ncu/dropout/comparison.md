# NCU Comparison: dropout

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'p': 0.5, 'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4}` | `{'tile': 2048, 'occupancy': 32}` |
| bf16 | `{'p': 0.5, 'n': 20971520}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8}` | `{'tile': 2048, 'occupancy': 16}` |
| fp32 | `{'p': 0.5, 'n': 20971520}` | `{'BLOCK_SIZE': 1024, 'num_warps': 8}` | `{'tile': 512, 'occupancy': 4}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 18.24 us | 68.12 % | 68.12 % | 40.74 % | 40.31 % | 30.65 % | 5.21 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 19.10 us | 65.65 % | 65.65 % | 39.69 % | 38.55 % | 31.93 % | 5.03 Tbyte/s | 128 | 24 register/thread | 0 byte/block | 0 byte/block | 21 block / 32 block |
| bf16 | triton | 18.53 us | 68.12 % | 68.12 % | 39.88 % | 39.71 % | 36.69 % | 5.21 Tbyte/s | 256 | 19 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| bf16 | cutile | 18.91 us | 66.33 % | 66.33 % | 39.58 % | 38.98 % | 33.67 % | 5.08 Tbyte/s | 128 | 27 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | triton | 34.43 us | 80.83 % | 80.83 % | 38.28 % | 42.45 % | 21.53 % | 6.20 Tbyte/s | 256 | 20 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 34.53 us | 80.88 % | 80.88 % | 37.15 % | 42.72 % | 37.99 % | 6.19 Tbyte/s | 128 | 16 register/thread | 0 byte/block | 0 byte/block | 32 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.05× faster** (18.2 µs vs 19.1 µs).
- **bf16**: Triton is **1.02× faster** (18.5 µs vs 18.9 µs).
- **fp32**: Triton is **1.00× faster** (34.4 µs vs 34.5 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Memory is more heavily utilized than Compute
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Memory is more heavily utilized than Compute
- **fp16 / triton** — Memory is more heavily utilized than Compute
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing DRAM in the Memory Workload Analysis section.
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
