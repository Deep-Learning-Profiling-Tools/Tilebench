# NCU Comparison: linear_self_attention

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0  
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.  

## Test cases (sweep-max per dtype)

| dtype | params | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|---|
| fp32 | `{'eps': 1e-06, 'M': 10000, 'D': 256}` | `{'BLOCK_M': 32, 'BLOCK_D': 32, 'num_warps': 1, 'num_stages': 1}` | `{'block_m': 64, 'block_d': 32, 'occupancy': 16}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Mem Tput % | DRAM % | L1 % | L2 % | Compute % | Mem BW | Block Sz | Regs | Static Shm | Dyn Shm | Blk Lim (R/S) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | triton | 5108.80 us | 99.55 % | 0.05 % | 99.92 % | 64.91 % | 11.61 % | 4.18 Gbyte/s | 32 | 29 register/thread | 0 byte/block | 0 byte/block | 64 block / 32 block |
| fp32 | cutile | 20052.01 us | 99.40 % | 0.01 % | 99.92 % | 50.94 % | 22.45 % | 1.05 Gbyte/s | 128 | 28 register/thread | 0 byte/block | 0 byte/block | 16 block / 32 block |

## Per-kernel breakdown (multi-kernel pipelines)

End-to-end Duration in the headline above sums every kernel launched per `impl.run()` call. This table lists each kernel in launch order; the headline rate metrics (Mem%, Compute%, etc.) come from the heaviest kernel of the pipeline.

| dtype | backend | k# | kernel duration | kernel name |
|---|---|---|---|---|
| fp32 | cutile | 1/3 | 19570.00 us | `_kv_kernel_Kt1_A2f32_1v4l0_2t1_3i16_p16_A2f32_1v4l` |
| fp32 | cutile | 2/3 | 230.59 us | `_z_kernel_Kt1_A1f32_1i16t1_p16_A2f32_1v4l0_2t1_3i1` |
| fp32 | cutile | 3/3 | 251.42 us | `_out_kernel_Kt1_A2f32_1v4l0_2t1_3i16_p16_A2f32_1v4` |
| fp32 | triton | 1/3 | 4900.00 us | `_kv_kernel` |
| fp32 | triton | 2/3 | 67.90 us | `_z_kernel` |
| fp32 | triton | 3/3 | 140.90 us | `_out_kernel` |

## Key findings (auto-derived)

- **fp32**: Triton is **3.92× faster** (5108.8 µs vs 20052.0 µs).

## NCU's own bottleneck verdict

- **fp32 / cutile** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.
- **fp32 / triton** — This workload is utilizing greater than 80.0% of the available compute or memory performance of this device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L1 in the Memory Workload Analysis section.

## Reports

- `cutile_fp32.ncu-rep`
- `triton_fp32.ncu-rep`

## Notes

Bottleneck verdicts above come from NCU's own SOLBottleneck rule (headline `OPT` recommendation). For per-section detail, open the .ncu-rep in `ncu-ui` or run `ncu --import <file> --page details | less`.
