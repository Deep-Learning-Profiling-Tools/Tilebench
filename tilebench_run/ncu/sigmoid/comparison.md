# NCU Comparison: sigmoid

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'n': 50000000}` | `{'BLOCK_SIZE': 4096, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 4096, 'occupancy': 16}` |
| bf16 | `{'n': 50000000}` | `{'BLOCK_SIZE': 2048, 'num_warps': 4, 'num_stages': 1}` | `{'tile': 4096, 'occupancy': 16}` |
| fp32 | `{'n': 50000000}` | `{'BLOCK_SIZE': 2048, 'num_warps': 8, 'num_stages': 1}` | `{'tile': 4096, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 30.43 us | 64.51 % | 64.51 % | 47.54 % | 35.81 % | 72.37 % | 4.94 Tbyte/s | 256 | 29 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp16 | cutile | 50.98 us | 40.00 % | 40.00 % | 27.22 % | 21.53 % | 73.85 % | 3.07 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | triton | 30.24 us | 64.85 % | 64.85 % | 47.73 % | 36.04 % | 73.44 % | 4.97 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| bf16 | cutile | 50.94 us | 40.05 % | 40.05 % | 27.23 % | 21.55 % | 74.04 % | 3.07 Tbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | triton | 55.49 us | 81.79 % | 81.79 % | 47.88 % | 41.06 % | 39.04 % | 6.27 Tbyte/s | 256 | 21 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 64.90 us | 70.80 % | 70.80 % | 41.90 % | 35.22 % | 57.11 % | 5.43 Tbyte/s | 128 | 30 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |

## Key findings (auto-derived)

- **fp16**: Triton is **1.68× faster** (30.4 µs vs 51.0 µs).
- **bf16**: Triton is **1.68× faster** (30.2 µs vs 50.9 µs).
- **fp32**: Triton is **1.17× faster** (55.5 µs vs 64.9 µs).

## NCU's own bottleneck verdict

- **bf16 / cutile** — Compute is more heavily utilized than Memory
- **bf16 / triton** — Compute and Memory are well-balanced
- **fp16 / cutile** — Compute is more heavily utilized than Memory
- **fp16 / triton** — Compute and Memory are well-balanced
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
