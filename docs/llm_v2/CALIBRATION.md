# CALIBRATION — empirical Roofline target of the TileBench++ LLM protocol

Decided direction (NEXT_STEP_EMPIRICAL_CALIBRATION.md, 2026-10-06): the
scoring ceiling of every task is an **empirically calibrated modelled
target** built from task-independent, device-measured sustained rates:

    T_emp     = max(F / P_emp[mode], Q / BW_emp)     (compute term only when the declared model has one)
    E_emp(B)  = max({T_emp / T_i : C_i <= B and valid_i} union {0})

`ceiling_basis = empirical` is recorded in every target. T_emp is not a
proof of the shortest achievable time: ratios above 1 are kept and flagged
for audit, never clipped or treated as a hacking verdict. Datasheet values
stay as a separate reference (`datasheet_reference`) and are never written
into an empirical profile. The old `B200.json` compute entries (datasheet
since PR #95) are not used for scoring any more.

## 1. Modes (calibration/schema.py `MODES`)

Every rate is registered under an explicit mode whose meaning is fixed:
input format, product/accumulation, unit. Names are matched exactly; there
is no alias. An IEEE fp32 GEMM is not TF32, an int8 matrix dot (OP/s) is
not an int32 vector rate, fp8 e4m3 is not e5m2 or AMD fnuz.

| mode | unit | probe (NVIDIA) | role |
|---|---|---|---|
| `hbm_stream_bw` | byte/s | D2D copy (`copy_`) and a Triton SM copy kernel (primary); SM read-reduce and fill (diagnostic) | scoring |
| `mma_fp16_f32acc`, `mma_bf16_f32acc` | FLOP/s | cuBLAS `torch.matmul`, reduced-precision reduction off | scoring |
| `mma_tf32_f32acc` | FLOP/s | cuBLAS `torch.matmul` with `fp32_precision = "tf32"` set and read back | scoring |
| `mma_fp8_e4m3_f32acc` | FLOP/s | cuBLASLt `torch._scaled_mm`, e4m3 operands, bf16 output, no fast-accum | scoring |
| `mma_int8_i32acc` | OP/s | cuBLASLt `torch._int_mm` | scoring |
| `fp32_fma_vector`, `fp16x2_fma_vector`, `bf16x2_fma_vector` | FLOP/s | Triton micro-benchmark: 8 independent FMA chains per lane, runtime trip count | scoring |
| `gemm_fp32_ieee` | FLOP/s | cuBLAS `torch.matmul` with `fp32_precision = "ieee"` | control only, never a ceiling |
| `int32_vector` | OP/s | no probe registered (never derived from an int8 matrix rate) | scoring (unavailable) |
| `mma_xf32_f32acc`, `mma_fp8_e4m3fnuz_f32acc` | FLOP/s | ROCm counterparts, measured on MI300X only | scoring |

Tasks may also declare `memory_only` (T = Q / BW, decision M2) or
`no_compute_term` (frozen F = 0).

## 2. Protocol (`tilebench-empirical-roofline/2`, calibration/protocol.py)

- Each point: 30 warm-up launches, then 3 independent batches of 100
  samples; a sample is a CUDA-event interval around `inner` back-to-back
  launches (`inner` calibrated so a sample lasts >= 2 ms). Point time =
  median over batches of the batch medians. Raw per-batch samples are kept
  in `raw/<mode>.json`.
- Mode value = highest throughput over the registered **valid** points.
  Never the fastest single sample. For bandwidth only primary probes at
  working sets >= 4 x LLC are eligible, cache-scale points are diagnostics,
  and the eligible points of the selected probe must form a plateau (spread
  <= 5 %). For compute modes a best point at the edge of the registered
  range that beats the runner-up by > 2 % is flagged
  `saturation_not_demonstrated`.
- Validity of a point: correctness check passes AND the point is
  device-bound: an untimed profiler pass over back-to-back launches shows
  no idle device time (busy share >= 0.95) or its per-launch device time
  explains the event-timed time (ratio >= 0.95). Dispatch-bound points
  (small GEMMs, 16–32 MiB copies) fail this and are excluded.
- Correctness / non-eliminable work: GEMM blocks compared against float64
  (exact int64 for int8); the relative error window also identifies the
  precision path (IEEE ~1e-6, TF32 ~3e-4). Copies compared in full, reads
  by per-block checksums, fills by a non-zero pattern on random bytes (no
  zero-page compression). FMA probes: closed-form expected output, 2x-loop
  scaling check, and the expected FMA opcodes counted in the SASS listing
  (`FFMA2`, `HFMA2`, `HFMA2.BF16`).
- Precision controls are set per mode through torch's `fp32_precision`
  API only (mixing it with the legacy `allow_tf32` getter makes torch
  refuse later matmuls) and the effective flags are recorded with the mode.
- Environment: device properties, driver, visible devices, MIG state,
  toolchain versions, BLAS libraries actually loaded (a mixed
  cuBLAS/cuBLASLt stack is refused), other compute processes (refused),
  and `nvidia-smi` telemetry (SM/memory clock, power, temperature,
  throttle reasons) every 250 ms during each mode; unreadable fields are
  `unavailable`.
- The calibration takes the same per-device lock as the evaluator
  (`outputs/llm_v2/locks/<device>.lock`): never concurrent with a campaign.
- GEMM shapes 4096 / 8192 / 12288 / 16384 (square); copies 16 MiB – 4 GiB;
  FMA sweeps 4 / 8 / 16 / 32 programs per SM.

## 3. Artifacts and commands

```
python -m tilebench.llm.v2 calibrate --device B200             # writes artifacts/llm_v2/calibration/B200/<id>/
python -m tilebench.llm.v2 calibration-check <dir>/profile.json
python -m tilebench.llm.v2 calibration-register --device B200 --profile <dir>/profile.json --status candidate
python -m tilebench.llm.v2 calibration-register --device B200 --profile <dir>/profile.json --status frozen --by "<owner>"
python -m tilebench.llm.v2 scoring-table --device B200 --out artifacts/llm_v2/scoring/B200/<file>.json
```

`calibrate` writes ONLY a new calibration directory (`protocol.json`,
`environment.json`, `raw/<mode>.json`, `profile.json` sealed with
`profile_sha256`, `summary.json`, `SHA256SUMS`). It refuses any output
under `tilebench/data/peak_performance/`, `results/` or the package, and
has no option to update the legacy table. `--quick` exercises the code
path; a quick profile can never be registered. On dgx003 the tilebench_env
needs `LD_LIBRARY_PATH` unset (the system cuBLASLt 13.4 would otherwise be
loaded under the wheel's cuBLAS 13.1 and cuBLASLt calls fail); the BLAS
stack check reports exactly this.

`manifests/calibration.yaml` names each device's active profile (path +
file sha256, status none / candidate / frozen). A campaign pins its own
device's entry and the declaration file sha at creation (`scoring_binding`
in `campaign_*.json` and every `trajectory.json`); a formal resume refuses
a changed profile or declaration; entries of other devices are not part of
the binding. The manifest is not part of the study config hash.

## 4. Declarations (`manifests/arithmetic_modes.yaml`, revision 2)

Modes are declared **per operator and dtype** from the contract's
precision section (no dtype-wide defaults): MMA operators use the matrix
mode of their operand format (fp32 inputs as TF32, as the contracts
state), non-matrix operators with mandated fp32 arithmetic use
`fp32_fma_vector`, operators allowed to compute in the input dtype use the
packed vector mode, and operators whose F is an element/comparison/
conversion count are proposed as `memory_only`. `f_kind` / `q_kind` say
what the frozen F and Q are (flop_count, dense_equivalent_flop,
simplified_op_count, int_op_count, element_count, zero; compulsory_io,
algorithm_specific). F/Q themselves are not rewritten; proposed overrides
(flash_attention causal F, radix_sort Q) sit under `decisions` and apply
only when approved. Until the declaration is approved every task's target
is `definition_pending` with a `provisional_t_emp_ms` for review.

## 5. B200 (dgx003, 2026-10-06, calibration `B200-20261006T063415Z-39b55bd3`)

Single whole device (no MIG), `CUDA_VISIBLE_DEVICES=0`, torch 2.10.0+cu130
/ CUDA 13.0 / triton 3.6.0 / cuBLAS 13.1.0.3, driver per
`environment.json`. Status: **frozen** on 2026-10-06 by the study owner
(FINAL_FREEZE_AND_START_B200.md §4 D2; `manifests/calibration.yaml`,
profile FILE sha256 `96a55a67…`, embedded seal `430660af…`); raw points,
telemetry, flags and seal are unchanged.

Sensitivity note (methodology): the selected `mma_tf32_f32acc` point
(M=4096, 664.5 TFLOP/s) carries the `points_batch_spread_flagged` audit flag
(batch spread 0.054 > 0.03). The next point, M=8192 (659.1 TFLOP/s, spread
0.007), is within 0.8 %; M=12288 / 16384 give 650.6 / 638.8. The frozen
value is kept as selected (max over valid points, never rewritten post
hoc); papers report the M=8192 value as a sensitivity check. The
`gemm_fp32_ieee` control carries `saturation_not_demonstrated`; it is not a
scoring mode.

| mode | measured | selected point | legacy 2026-04-08 | datasheet (B200.json) |
|---|---|---|---|---|
| `hbm_stream_bw` | 6840.4 GB/s | SM copy, 4 GiB (D2D copy plateau 6.43 TB/s) | 6539.4 GB/s (D2D copy, 1 GiB) | 7700 (not in file) |
| `mma_fp16_f32acc` | 1267.2 TFLOP/s | M=12288 | 1335.6 | 2250 |
| `mma_bf16_f32acc` | 1341.4 TFLOP/s | M=12288 | 1394.6 | 2250 |
| `mma_tf32_f32acc` | 664.5 TFLOP/s | M=4096 | not measured | 1100 |
| `mma_fp8_e4m3_f32acc` | 2588.4 TFLOP/s | M=16384 | 2629.9 | 4500 |
| `mma_int8_i32acc` | 2721.4 TOP/s | M=16384 | 2742.0 (labelled TFLOPS) | 4500 |
| `fp32_fma_vector` | 70.0 TFLOP/s | 32 programs/SM | — (66.5 was SGEMM) | — |
| `fp16x2_fma_vector` | 73.9 TFLOP/s | 32 programs/SM | — | — |
| `bf16x2_fma_vector` | 73.9 TFLOP/s | 32 programs/SM | — | — |
| `gemm_fp32_ieee` (control) | 66.5 TFLOP/s | M=16384, flagged saturation_not_demonstrated | 66.5 | — |

Observations recorded in the raw files: during every GEMM mode the SM
clock dropped from 1965 MHz to about 1000 MHz with power at the 1000 W
limit, so the matrix rates are power-capped sustained rates, as intended;
the FMA probes ran at 1965 MHz. The three fp32 numbers differ because
they measure different things (SGEMM library rate, TF32 tensor-core
rate, scalar FMA rate) and none of them is "the fp32 peak". The SM copy
kernel exceeds the D2D copy engine path by 6 %; both are kept, the
primary maximum is taken over both.

## 6. Other devices (same code, local measurement, no scaling from B200)

- **GH200** (`exp/gh200` window): `python -m tilebench.llm.v2 calibrate --device GH200`
  with the same commit; register as candidate; the GH200 profile does not
  touch B200's entry.
- **MI300X**: the ROCm backend is detected from `torch.version.hip`; the
  same probes run through hipBLAS/hipBLASLt (`mma_xf32_f32acc` replaces TF32,
  `mma_fp8_e4m3fnuz_f32acc` replaces e4m3) and the Triton AMDGCN listing is
  checked for `v_pk_fma_*` / `v_fma_f32`. Not yet exercised on hardware;
  the first run there must verify `/dev/kfd` + `/dev/dri`, the LLC value
  used for HBM eligibility (`tilebench.hardware.last_level_cache_bytes`),
  the precision controls hipBLASLt honours, and that `rocm-smi` telemetry
  is parsed. Fix shared code on `exp/llm`; do not fork the schema.
- **Trn2**: `calibrate --device Trn2` is refused by the shared code on
  purpose. The native adapter is written in the authorized NKI window
  against the same profile schema: fix the actual Neuron device, LNC
  configuration, visible logical/physical cores, launch grid and HBM
  address space first; measure on the same scope the formal tasks use
  (never a chip/instance rate divided by a core count, never two copies
  of one output counted twice); time real inputs with complete device
  execution and prove it (no synthetic-input default benchmark, no
  host-transfer interval); check the installed `nki` API version, not a
  web page. Register under the same `calibration.yaml` entry.

## 7. Freeze and formal gates

A formal campaign on a device needs: `calibration.yaml` entry `frozen`
(owner), `arithmetic_modes.yaml` `approved` (decision M1) with every
in-scope task `ok` (no pending decision, no unavailable peak), plus the
existing gates (approved contracts, approved/frozen skills, approved
models, provider grants, sandbox, host arch). `preflight --live --run-type
formal` lists each missing item; `scoring-table` shows the per-task
status. Same device = one profile for Base and Enhanced, every DSL and
every model.
