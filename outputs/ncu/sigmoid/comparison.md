# NCU Comparison: sigmoid

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 50000000}` | `{'BLOCK_SIZE': 4096, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 4096, 'occupancy': 16}` |
| bf16 | `{'n': 50000000}` | `{'BLOCK_SIZE': 4096, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 4096, 'occupancy': 16}` |
| fp32 | `{'n': 50000000}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 2048, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 34.34 us | 76.25 % | 76.25 % | 41.00 % | 34.96 % | 65.13 % | 5.84 Tbyte/s | 256 | 29 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp16 | cutile | 52.00 us | 53.16 % | 53.16 % | 26.72 % | 23.44 % | 72.87 % | 4.07 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | triton | 34.24 us | 76.65 % | 76.65 % | 41.81 % | 35.04 % | 65.59 % | 5.88 Tbyte/s | 256 | 30 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| bf16 | cutile | 51.39 us | 53.83 % | 53.83 % | 26.69 % | 23.73 % | 72.88 % | 4.13 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | triton | 63.10 us | 83.32 % | 83.32 % | 41.68 % | 37.97 % | 34.68 % | 6.39 Tbyte/s | 256 | 21 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 65.06 us | 81.52 % | 81.52 % | 41.41 % | 36.92 % | 61.10 % | 6.25 Tbyte/s | 128 | 40 register/thread | 0 byte/block | 0 byte/block | 12 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.51× faster** (34.3 µs vs 52.0 µs).
- **bf16**: Triton is **1.50× faster** (34.2 µs vs 51.4 µs).
- **fp32**: Triton is **1.03× faster** (63.1 µs vs 65.1 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Compute is more heavily utilized than Memory
- **bf16 / triton** — Memory is more heavily utilized than Compute
- **fp16 / cutile** — Compute is more heavily utilized than Memory
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
