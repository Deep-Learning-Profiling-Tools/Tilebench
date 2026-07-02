# NCU Comparison: 3d_conv

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input.

**Operator:** multi-channel Conv3d forward (batch=1, C_in=C_out=64, k=3x3x3, pad=1) — **implicit GEMM** on both
backends: one CTA computes a (BLOCK_M x BLOCK_OUT) output tile, contracting over C_in*k^3 = 1728
via Tensor-Core MMA (Triton `tl.dot(input_precision="tf32")`, cuTile `ct.mma` with tfloat32/native
cast). torch reference: native-dtype cuDNN (fp16 -> fp16 TC, fp32 -> TF32 where cuDNN supports it).
Same algorithm, same tiling scheme, same autotune tile space ({32,64,128}x{16,32,64}x{64,128}) on
both backends; DSL-specific knobs differ (Triton num_warps/num_stages vs cuTile occupancy).

## Autotune winners (sweep-max, H=320 (D=32))

| dtype | Triton | cuTile |
|---|---|---|
| fp16 | `{'BLOCK_SIZE_BATCH_DHW': 128, 'BLOCK_SIZE_IN_FEAT': 64, 'BLOCK_SIZE_OUT_FEAT': 64, 'num_warps': 4, 'num_stages': 3}` | `{'block_bdhw': 128, 'block_in': 16, 'block_out': 64, 'occupancy': 4}` |
| fp32 | `{'BLOCK_SIZE_BATCH_DHW': 64, 'BLOCK_SIZE_IN_FEAT': 64, 'BLOCK_SIZE_OUT_FEAT': 64, 'num_warps': 4, 'num_stages': 2}` | `{'block_bdhw': 128, 'block_in': 16, 'block_out': 64, 'occupancy': 4}` |

## Headline (autotune-best @ sweep-max; times from the sweep CSV, ms from NCU)

| dtype | Backend | NCU duration | Tensor pipe % | UTCMMA (tcgen05) | Occupancy % | SM % |
|---|---|---|---|---|---|---|
| fp16 | triton | 13.9 ms | 2.3 | 2,764,800 | 12.5 | 67.5 |
| fp16 | cutile | 35.8 ms | 0.9 | 2,764,800 | 31.1 | 71.1 |
| fp32 | triton | 18.2 ms | 7.1 | 11,059,200 | 18.7 | 63.8 |
| fp32 | cutile | 33.2 ms | 2.0 | 5,529,600 | 31.1 | 76.1 |

## Sweep-max latencies (autotune CSV)

| dtype | torch_ms | triton_ms | cutile_ms | C/T |
|---|---|---|---|---|
| fp16 | 1.4217 | 13.8831 | 35.8464 | 2.5820 |
| fp32 | 2.0816 | 18.2222 | 33.1941 | 1.8216 |

## Key findings

- **Both backends run on Tensor Cores** (tensor pipe active on every report). Triton compiles to
  tcgen05 `UTCMMA`; cuTile picks tcgen05 or legacy HMMA depending on the tile shape (UTCMMA=0
  rows still show a busy hmma sub-pipe).
- Both DSLs trail torch's cuDNN, which uses dedicated implicit-GEMM conv kernels; the DSL kernels
  spend most cycles on im2col index arithmetic (ALU-bound, SM% 50-77 with low tensor%), identically
  on both sides — so the Triton-vs-cuTile delta isolates DSL codegen, which is the benchmark's goal.

## Reports

- `triton_fp16.ncu-rep`, `triton_fp32.ncu-rep`, `cutile_fp16.ncu-rep`, `cutile_fp32.ncu-rep`
