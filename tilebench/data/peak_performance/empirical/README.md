# Empirical peak performance (TileArena)

These device-specific summaries contain **measured sustained throughput and bandwidth**, not vendor datasheet theoretical peaks. They are copied from frozen TileArena empirical calibration profiles (protocol `tilebench-empirical-roofline/2`). The values are used for the empirical roofline/SOL target through the frozen calibration profiles, **not** by reading these summary JSON files.

| Summary | Frozen calibration ID | Source availability |
| --- | --- | --- |
| `B200.json` | `B200-20261006T063415Z-39b55bd3` | Full source profile available in `main` |
| `GH200.json` | `GH200-20261007T215924Z-0a3f4803-int8layout` | Frozen source profile not yet published on the remote device branch |
| `MI300X.json` | `MI300X-20261009T060617Z-eccfca2c` | Frozen source profile not yet published on the remote device branch |

The complete calibration evidence is organized under `artifacts/llm_v2/calibration/<device>/<calibration_id>/`: `profile.json`, original measurements, environment, protocol, summary, and hashes. Each summary records the exact source profile path and SHA-256 for future verification. GH200 and MI300X source files will be published separately; the summaries do **not** themselves contain all raw evidence.

## Data fields

- `schema: tilearena-empirical-peak/1`
- `modes`: measured `hbm_stream_bw` and arithmetic modes; `value` uses exact source SI units (`byte/s`, `FLOP/s`, or `OP/s`), while `value_scaled` is for convenient reading.
- `modes_not_calibrated`: unavailable/failed modes with **no measured value**.
- `role: scoring` identifies performance ceilings; `role: control` is a diagnostic comparison and **not** a SOL ceiling.
- `calibration_id` and `source` provide provenance, profile hashes, and measurement context.

Matrix throughput depends on device, arithmetic mode, and measured path: verified BLAS **or Triton** kernels are used. For example, the GH200 INT8 profile corrects an unfavorable input layout; MI300X INT8 uses a verified Triton GEMM probe. AMD FP8 `e4m3fnuz` differs from NVIDIA `e4m3fn` and must not be conflated.

These are sustained measured rates and **not provable global hardware upper bounds**: suitably optimized kernels or other shapes may occasionally exceed them. SOL efficiencies above 1 are preserved and audited rather than clipped.

The legacy `../B200.json` and `../Trainium2.json` are unchanged. In particular, the legacy B200 table combines datasheet compute values and measured bandwidth, so it must not be substituted for the frozen TileArena empirical calibration. Existing consumers of `peak_performance/<device>.json` are unaffected.
