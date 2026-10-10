# Empirical peak performance

These are sustained rates measured by TileArena's empirical roofline calibration, **not vendor theoretical
(datasheet) peaks**. Each file lists only the modes that were successfully measured on that device. Units:
`peak_bw_GBs` in GB/s, `peak_tflops` in TFLOP/s, `peak_tops` (INT8) in TOP/s.

FP16 compute peaks:
- `fp16_mma`: FP16 matrix operations on Tensor Cores / Matrix Cores (MMA).
- `fp16_vector`: FP16 arithmetic without MMA. The value is the measured packed FP16x2 FMA throughput (calibration
  mode `fp16x2_fma_vector`). FP16x2 is only the packed execution path used for the measurement, not a different
  precision; scalar FP16 was not measured separately.
- `fp32_vector`: non-MMA operations that require FP32 arithmetic.

BF16 uses the same naming (`bf16_mma`, `bf16_vector` from packed BF16x2 FMA). MI300X has no valid packed BF16
measurement, so it has no `bf16_vector`.

The raw calibration data of each device is in `artifacts/llm_v2/calibration/<device>/<calibration_id>/`. For B200 it
is in this repository; the GH200 and MI300X runs are not yet published.
