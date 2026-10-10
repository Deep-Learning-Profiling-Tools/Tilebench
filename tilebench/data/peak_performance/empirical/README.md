# Empirical peak performance (TileArena)

These device-specific summaries contain **measured sustained throughput and bandwidth**, not vendor datasheet theoretical peaks. They are copied from frozen TileArena empirical calibration profiles (protocol `tilebench-empirical-roofline/2`). The values are used for the empirical roofline/SOL target through the frozen calibration profiles, **not** by reading these summary JSON files.

| Summary | Frozen calibration ID | Source profile (path, file SHA-256) | Source availability |
| --- | --- | --- | --- |
| `B200.json` | `B200-20261006T063415Z-39b55bd3` | `artifacts/llm_v2/calibration/B200/B200-20261006T063415Z-39b55bd3/profile.json`<br>`96a55a674b64b40c1d29d23fd83394433df3254598fcf4f7112a533db2378502` | In `main` (added by `04d7e455`, #320); can be checked on GitHub |
| `GH200.json` | `GH200-20261007T215924Z-0a3f4803-int8layout` | `artifacts/llm_v2/calibration/GH200/GH200-20261007T215924Z-0a3f4803-int8layout/profile.json`<br>`d5d02cbd1d39c21d48c405d60a3e84043d1623749c6b2643621d1e856d64b4f7` | Committed only in the GH200 host checkout of `exp/llm-gh200` (`49f9a535`, `436d033f`); the remote `exp/llm-gh200` (`eccfca2c`) does not contain it, so it **cannot yet be checked on GitHub** |
| `MI300X.json` | `MI300X-20261009T060617Z-eccfca2c` | `artifacts/llm_v2/calibration/MI300X/MI300X-20261009T060617Z-eccfca2c/profile.json`<br>`d0e2604f071db7349fffda7f59df2f87727e5713db42e356d08e44faf2d38dc1` | Committed only in the MI300X host checkout of `exp/llm-mi300x` (`1e4fa1f4`); not pushed, so it **cannot yet be checked on GitHub** |

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
