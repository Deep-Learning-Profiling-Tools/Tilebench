# Empirical peak performance

One JSON file per device. Each file holds the sustained rates that TileArena measured on that device with its
empirical roofline calibration (protocol `tilebench-empirical-roofline/2`). The rates were copied unchanged from the
device's frozen calibration profile.

These are **measured sustained rates, not vendor datasheet theoretical peaks.**

- **Matrix modes:** each rate is what the vendor BLAS sustains on large square GEMMs.
- **Vector modes:** each rate is what a register-resident FMA micro-benchmark sustains.
- **Bandwidth:** the rate of streaming copy probes over working sets far larger than the last-level cache.

A kernel can therefore come close to, or slightly exceed, one of these rates.

| file | device | calibration |
|---|---|---|
| `B200.json` | NVIDIA B200 | `B200-20261006T063415Z-39b55bd3` (frozen) |

GH200 and MI300X files will be added in the same format from their frozen profiles.

## Relation to `../B200.json`

`tilebench/data/peak_performance/B200.json` is the older table. It combines datasheet dense compute peaks with a copy
bandwidth measured on 2026-04-08. That is a different definition, so both files are kept. Code that reads
`peak_performance/<device>.json` is not affected by this directory.

## Raw measurements

Every value here can be traced to the complete calibration run in
`artifacts/llm_v2/calibration/<device>/<calibration_id>/`:

| file | content |
|---|---|
| `profile.json` | all measured points, plateau checks and telemetry |
| `raw/` | raw probe measurements |
| `summary.json`, `environment.json`, `protocol.json`, `SHA256SUMS` | run summary, environment, protocol and checksums |

The active profile of each device and its SHA-256 are registered in `tilebench/llm/v2/manifests/calibration.yaml`.
The `source` block of each JSON records:
- the profile path;
- the profile file's SHA-256;
- the profile's internal seal (`profile_sha256`).

## Fields

| field | content |
|---|---|
| `memory_bandwidth` | measured bandwidth (mode `hbm_stream_bw`, `byte/s`) |
| `arithmetic_modes` | every calibrated mode under its exact profile name (e.g. `mma_fp16_f32acc`, `fp32_fma_vector`) |
| `not_measured_modes` | modes this calibration did not measure, with the profile's reason; they have no value |

**Per-mode fields** (in `memory_bandwidth` and in each `arithmetic_modes` entry):
- `value` and `unit` are the profile's SI values: `FLOP/s`, `OP/s` (integer) or `byte/s`.
- `value_scaled` and `unit_scaled` give the same value in `TFLOP/s`, `TOP/s` or `GB/s`, rounded to 2 decimals.
- `role: scoring` marks rates used for the modelled target `T_SOL = max(F / P_mode, Q / BW)`. Each operator and dtype
  declares its mode in `tilebench/llm/v2/manifests/arithmetic_modes.yaml`.
- `role: control` (`gemm_fp32_ieee`, an IEEE-FP32 library SGEMM rate) is a reference only, never a ceiling.
- `flags` and `limitations` are copied from the profile.
