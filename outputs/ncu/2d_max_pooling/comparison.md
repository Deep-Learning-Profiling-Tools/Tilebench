# NCU Comparison: 2d_max_pooling

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'N': 4, 'C': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'H': 640}` | `{'BLOCK_R': 1, 'BLOCK_C': 512, 'num_warps': 4}` | `{'tile_r': 4, 'tile_c': 128, 'occupancy': 8}` |
| bf16 | `{'N': 4, 'C': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'H': 640}` | `{'BLOCK_R': 1, 'BLOCK_C': 512, 'num_warps': 4}` | `{'tile_r': 4, 'tile_c': 128, 'occupancy': 8}` |
| fp32 | `{'N': 4, 'C': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'H': 640}` | `{'BLOCK_R': 1, 'BLOCK_C': 512, 'num_warps': 4}` | `{'tile_r': 4, 'tile_c': 128, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 203.36 us | 55.91 % | 36.35 % | 57.04 % | 19.88 % | 77.04 % | 2.79 Tbyte/s | 128 | 44 register/thread | 0 byte/block | 1.02 Kbyte/block | 10 block / 32 block |
| fp16 | cutile | 290.72 us | 86.02 % | 29.38 % | 87.24 % | 17.14 % | 80.86 % | 2.25 Tbyte/s | 128 | 64 register/thread | 1.04 Kbyte/block | 0 byte/block | 8 block / 30 block |
| bf16 | triton | 205.12 us | 55.47 % | 36.05 % | 56.63 % | 19.70 % | 85.61 % | 2.77 Tbyte/s | 128 | 45 register/thread | 0 byte/block | 1.02 Kbyte/block | 10 block / 32 block |
| bf16 | cutile | 290.66 us | 86.27 % | 29.39 % | 87.33 % | 17.14 % | 81.10 % | 2.25 Tbyte/s | 128 | 64 register/thread | 1.04 Kbyte/block | 0 byte/block | 8 block / 30 block |
| fp32 | triton | 227.52 us | 77.25 % | 62.48 % | 78.79 % | 35.01 % | 63.45 % | 4.79 Tbyte/s | 128 | 39 register/thread | 0 byte/block | 2.05 Kbyte/block | 12 block / 33 block |
| fp32 | cutile | 306.43 us | 86.78 % | 50.09 % | 88.03 % | 26.05 % | 76.08 % | 3.84 Tbyte/s | 128 | 64 register/thread | 2.06 Kbyte/block | 0 byte/block | 8 block / 20 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.43× faster** (203.4 µs vs 290.7 µs).
- **bf16**: Triton is **1.42× faster** (205.1 µs vs 290.7 µs).
- **fp32**: Triton is **1.35× faster** (227.5 µs vs 306.4 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **bf16 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **fp16 / triton** — Compute is more heavily utilized than Memory
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
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
