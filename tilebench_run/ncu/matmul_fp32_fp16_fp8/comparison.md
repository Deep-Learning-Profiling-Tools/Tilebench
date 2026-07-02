# NCU Comparison: matmul_fp32_fp16_fp8

**Hardware:** NVIDIA B200 180GB (dgx003), CUDA 13, NCU 2026.1.1.0
**Profile method:** `--set full --import-source on`, `--launch-skip 3 --launch-count 1`, autotune-winner cfg at sweep-max input (`M=N=4096, K=20480`).

**Operator state:** Triton uses **host-side TMA** (`TensorDescriptor` + autotune
`pre_hook`, tutorial-09 pattern; non-persistent, grouped swizzle, TF32 for fp32,
fp32 accumulate) — TMA parity with cuTile, whose codegen uses TMA implicitly.
`fp8_e5m2` was dropped from the operator (torch/cuBLASLt reject e5m2×e5m2, so no
torch-native baseline exists). torch baseline: TF32 for fp32, `torch._scaled_mm`
(real fp8 Tensor-Core GEMM) for fp8_e4m3fn.

## Test cases (sweep-max per dtype)

| dtype | autotune cfg (Triton) | autotune cfg (cuTile) |
|---|---|---|
| fp32 | `{BM:128, BN:128, BK:32, GS:8, warps:4, stages:3}` | `{tm:256, tn:256, tk:64, gs:8, occupancy:4}` |
| fp16 | `{BM:256, BN:256, BK:64, GS:8, warps:4, stages:3}` | `{tm:256, tn:256, tk:64, gs:8, occupancy:8}` |
| fp8_e4m3fn | `{BM:256, BN:256, BK:128, GS:8, warps:4, stages:3}` | `{tm:256, tn:256, tk:128, gs:8, occupancy:8}` |

## Headline (per dtype, both backends)

| dtype | Backend | Duration | Tensor pipe % | Occupancy % | vs pre-TMA Triton |
|---|---|---|---|---|---|
| fp32 | triton | 1.33 ms | 60.0 | 12.0 | was 6.13 ms / 10.2% tensor |
| fp32 | cutile | 870 µs | 96.6 | 10.9 | |
| fp16 | triton | 518 µs | 85.7 | 6.3 | was 672 µs / 63.5% |
| fp16 | cutile | 467 µs | 97.2 | 10.9 | |
| fp8_e4m3fn | triton | 257 µs | 82.1 | 6.2 | was 576 µs / 34.0% |
| fp8_e4m3fn | cutile | 221 µs | 95.6 | 10.9 | |

## Key findings

- **cuTile/Triton gap is now 1.1–1.5×** (fp32 1.53×, fp16 1.11×, fp8 1.17×) —
  down from 6.25× / 1.30× / 2.49× before the Triton TMA rewrite.
- **The fp32 anomaly is resolved.** The old pointer+smem-pipelined kernel
  collapsed at fp32: 4-byte tiles → 98 KB smem + 123 regs → 25% occupancy →
  0.15 eligible warps/scheduler → tensor pipe 90% idle (10.2%). With TMA the
  hardware copy engine hides latency **without occupancy**: occupancy actually
  *dropped* to 6–12% while the tensor pipe rose to 60–86% — the same design
  point cuTile has always operated at (≈11% occupancy, 95–97% tensor).
- Both backends issue identical TF32/HMMA UTCMMA work (tcgen05); the remaining
  gap is codegen quality: cuTile keeps the tensor pipe ~97% fed, Triton 60–86%.
  fp32 is the widest because Triton's smem budget only admits a 128×128×32 tile
  (4-byte elements), while cuTile runs 256×256×64.

## Implementation notes forced by TMA (see impl_triton.py)

- **B is consumed transposed (N, K)** and cached per input tensor: a
  `[BLOCK_K, BLOCK_N]` box on row-major (K, N) B has a 256–512 B inner dim,
  past the 128 B TMA-swizzle fast path, and regressed fp16/fp8.
- **`DT_ID` constexpr in the autotune key**: `TensorDescriptor` args are not
  `torch.Tensor`s, so the autotuner stops folding dtypes into its cache key —
  without it, every dtype inherited the first dtype's winner.
- **fp32-viable configs added** (`128×128`, `bk=64/ns2`, `bk=32/ns3-4`): with
  TMA store staging, every original ns=3 config exceeds B200's 227 KB smem at
  4-byte elements and the autotuner crashed on OutOfResources.

## Reports

- `triton_fp32.ncu-rep`, `triton_fp16.ncu-rep`, `triton_fp8_e4m3fn.ncu-rep`
- `cutile_fp32.ncu-rep`, `cutile_fp16.ncu-rep`, `cutile_fp8_e4m3fn.ncu-rep`

## Notes

Open a `.ncu-rep` in `ncu-ui` or `ncu --import <file> --page details` for
per-section detail.
