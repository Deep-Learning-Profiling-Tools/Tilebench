# NCU Comparison: 1d_conv

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp16 | `{'kernel_size': 127, 'input_size': 20000000}` | `{'BLOCK_SIZE': 512, 'num_warps': 8, 'num_stages': 2}` | `{'tile': 2048, 'occupancy': 8}` |
| fp32 | `{'kernel_size': 127, 'input_size': 20000000}` | `{'BLOCK_SIZE': 256, 'num_warps': 8, 'num_stages': 2}` | `{'tile': 2048, 'occupancy': 8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | triton | 531.23 us | 83.51 % | 1.00 % | 84.38 % | 1.89 % | 93.12 % | 76.99 Gbyte/s | 256 | 24 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp16 | cutile | 597.76 us | 52.54 % | 1.44 % | 53.44 % | 1.55 % | 79.06 % | 110.52 Gbyte/s | 128 | 46 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | triton | 627.84 us | 95.47 % | 2.46 % | 96.34 % | 2.03 % | 91.20 % | 188.90 Gbyte/s | 256 | 18 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |
| fp32 | cutile | 451.17 us | 68.76 % | 3.34 % | 70.16 % | 2.21 % | 79.00 % | 255.94 Gbyte/s | 128 | 46 register/thread | 0 byte/block | 0 byte/block | 10 block / 32 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| fp16 | cutile | 1/2 | 579.94 us | `_conv1d_stencil_kernel_Kt1_A1f16_1i16t1_p16_A1f16_` |
| fp16 | cutile | 2/2 | 17.82 us | `void at::vectorized_elementwise_kernel<8, at::floa` |

## Key findings (auto-derived)

- **fp16**: Triton is **1.13× faster** (531.2 µs vs 597.8 µs).
- **fp32**: cuTile is **1.39× faster** (451.2 µs vs 627.8 µs).

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
