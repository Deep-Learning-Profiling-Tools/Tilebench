# Empirical peak performance

These are sustained rates measured by TileArena's empirical roofline calibration, **not vendor theoretical
(datasheet) peaks**. Each file lists only the modes that were successfully measured on that device. Units:
`peak_bw_GBs` in GB/s, `peak_tflops` in TFLOP/s, `peak_tops` (INT8) in TOP/s.

The raw calibration data of each device is in `artifacts/llm_v2/calibration/<device>/<calibration_id>/`. For B200 it
is in this repository; the GH200 and MI300X runs are not yet published.
