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
| fp16 | triton | 196.86 us | 57.58 % | 33.83 % | 58.86 % | 19.96 % | 79.29 % | 2.60 Tbyte/s | 128 | 44 register/thread | 0 byte/block | 1.02 Kbyte/block | 10 block / 32 block |
| fp16 | cutile | 286.18 us | 86.99 % | 27.27 % | 88.29 % | 17.44 % | 81.76 % | 2.09 Tbyte/s | 128 | 64 register/thread | 1.04 Kbyte/block | 0 byte/block | 8 block / 30 block |
| bf16 | triton | 199.52 us | 56.93 % | 33.36 % | 58.24 % | 19.66 % | 87.82 % | 2.56 Tbyte/s | 128 | 45 register/thread | 0 byte/block | 1.02 Kbyte/block | 10 block / 32 block |
| bf16 | cutile | 286.24 us | 86.87 % | 27.26 % | 88.30 % | 17.44 % | 81.64 % | 2.09 Tbyte/s | 128 | 64 register/thread | 1.04 Kbyte/block | 0 byte/block | 8 block / 30 block |
| fp32 | triton | 218.50 us | 80.34 % | 61.74 % | 82.11 % | 35.97 % | 65.90 % | 4.74 Tbyte/s | 128 | 39 register/thread | 0 byte/block | 2.05 Kbyte/block | 12 block / 33 block |
| fp32 | cutile | 297.86 us | 88.78 % | 49.06 % | 90.14 % | 26.84 % | 77.79 % | 3.76 Tbyte/s | 128 | 64 register/thread | 2.06 Kbyte/block | 0 byte/block | 8 block / 20 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.45× faster** (196.9 µs vs 286.2 µs).
- **bf16**: Triton is **1.43× faster** (199.5 µs vs 286.2 µs).
- **fp32**: Triton is **1.36× faster** (218.5 µs vs 297.9 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **bf16 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp16 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **fp16 / triton** — Compute is more heavily utilized than Memory
- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.

## Reports

- `cutile_bf16.ncu-rep`
- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_bf16.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
