# NCU Comparison: 3d_conv

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'input_depth': 64, 'kernel_depth': 3, 'kernel_rows': 3, 'kernel_cols': 3, 'input_rows': 640}` | `{'BLOCK_SIZE': 512, 'num_warps': 4}` | `{'tile': 512, 'occupancy': 8}` |
| fp32 | `{'input_depth': 64, 'kernel_depth': 3, 'kernel_rows': 3, 'kernel_cols': 3, 'input_rows': 640}` | `{'BLOCK_SIZE': 256, 'num_warps': 4}` | `{'tile': 512, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 264.03 us | 83.73 % | 5.58 % | 85.46 % | 9.68 % | 90.66 % | 428.24 Gbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp16 | cutile | 251.17 us | 43.41 % | 3.73 % | 44.21 % | 11.65 % | 78.00 % | 285.90 Gbyte/s | 128 | 64 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |
| fp32 | triton | 235.84 us | 94.43 % | 9.48 % | 96.34 % | 16.82 % | 76.86 % | 727.04 Gbyte/s | 128 | 32 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |
| fp32 | cutile | 216.51 us | 52.16 % | 10.37 % | 53.37 % | 19.59 % | 74.69 % | 795.40 Gbyte/s | 128 | 64 register/thread | 0 byte/block | 0 byte/block | 8 block / 32 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| fp16 | triton | 1/2 | 242.59 us | `_conv3d_kernel` |
| fp16 | triton | 2/2 | 21.44 us | `void at::vectorized_elementwise_kernel<8, at::floa` |

## Key findings (auto-derived)

- **fp16**: cuTile is **1.05× faster** (251.2 µs vs 264.0 µs).
- **fp32**: cuTile is **1.09× faster** (216.5 µs vs 235.8 µs).

## NCU's own bottleneck verdict

- **fp16 / cutile** — Compute is more heavily utilized than Memory
- **fp16 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the Compute Workload Analysis section.
- **fp32 / cutile** — Compute is more heavily utilized than Memory
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.

## Reports

- `cutile_fp16.ncu-rep`
- `cutile_fp32.ncu-rep`
- `triton_fp16.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
