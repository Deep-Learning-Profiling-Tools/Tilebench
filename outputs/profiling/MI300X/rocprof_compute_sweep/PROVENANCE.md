# MI300X ROCm Compute Profiler reports: provenance

Device: AMD Instinct MI300X (gfx942, CDNA3), ROCm 7.14.0, rocprof-compute 3.7.0 (git 418cd5f).
Methodology (every pair): `rocprof-compute profile` with the full default counter set (13 counter
passes plus roofline), selected by a start-anchored kernel-name regex. The kernels are checked against
the exact launch sequence in `outputs/profiling/MI300X/kernel_counts.json`. PC sampling runs as a
separate pass (stochastic, interval 65536).

Harness: one prime process (3 calls, no profiler), then the profiled process. That process does input
generation, a 512 MiB (2x LLC) eviction and exactly ONE `impl.run()`.

Inputs: the sweep-max case of each operator/dtype. Configs: the formal MI300X autotune winner from
`ncu_catalogue.json`, built from `results/MI300X/logs/autotune_logs/*_autotune_triton.json`.

Coverage: 109/109 valid Triton pairs. matmul_fp32_fp16_fp8 / fp8_e4m3fn is excluded: it is
UNSUPPORTED_DTYPE on MI300X and was not replaced by another dtype. Reports are in the Hugging Face
dataset `bcui2/NCU_report`, folder `AMD_MI300X/`, at revision
`645591bbb808b4ceeecae2c4ec61b78151ad6249`. `report_manifest.json` lists, for every pair, the
profiling source, a content hash of the pair directory and the HF commit it was uploaded in.

| reports | profiling source SHA | profiled | HF commit |
|---|---|---|---|
| 101 pairs | 5b82f8afed59f83c3f39696338e7ce51dec38de7 | 2026-10-02 | ff3600e1831dc3bed6925f301214cad6db8234cb |
| 8 pairs (replacements, below) | 124fdc948c322b591018adc00ddd5be71bc26289 | 2026-10-04 | f61b73dc, 69852499, 8b51e7ee, 645591bb |

The formal performance CSVs (`results/MI300X/csv/`, commit 5e714117) were measured on earlier
sources: default 005ab63b26e01a2124c666f7f66e405f3db547c0 and autotune
4d08985ae819e9063ae40e7cddd2af2451f458a6. Their provenance is unchanged. Between 4d08985a and
5b82f8af only profiling infrastructure changed. 124fdc94 changes three operators, listed below,
without changing any default or autotune behaviour.

## Replaced reports (2026-10-04)

A runtime audit compared, for all 109 pairs, every Triton launch of the profiling replay with the
formal autotune path restricted to the logged winner (`winner_replay_audit_pre_fix.csv`). It found
8 pairs whose profiled launch sequence was not the formal one. Their reports were re-profiled from
124fdc94 with the unchanged method and replaced on HF. The superseded versions remain at HF revision
ff3600e1; their content hashes are under `supersedes` in `report_manifest.json`.

- **batched_matmul fp16/bf16/fp32, streamk_matmul fp16/bf16/fp32: the architecture fallback
  overrode the winner.**
  - **Cause:** on CDNA3, `_default_config()` returned `_CDNA3_DEFAULT_CONFIG` unconditionally, so the
    winner replayed into `_DEFAULT_CONFIG` was never read. The old reports therefore profiled the CDNA3
    legality fallback. For example, batched_matmul fp16 ran workgroup 256 / LDS 16384 instead of the
    winner's 512 / 32768, and streamk_matmul fp32 ran a full_tiles grid of 13984 instead of 6688.
  - **Fix:** `_default_config()` now applies the fallback only while `_DEFAULT_CONFIG` is still the
    builtin default (`_BUILTIN_DEFAULT_CONFIG` sentinel). The precedence is: replayed winner > CDNA3
    fallback > builtin default.
- **bitonic_sort fp16/fp32: the pad_kernel replay config was inconsistent with the formal path.**
  - **Cause:** the untuned pad_kernel read `_DEFAULT_CONFIG["BLOCK"]` in both paths. The replayed
    winner (BLOCK 2048 / 4096) therefore also changed the pad launch's block and grid, while the formal
    autotune path pads with BLOCK 1024. The 300 tuned bitonic_step launches were already exact.
  - **Fix:** pad_kernel uses a fixed `_PAD_BLOCK = 1024`.

Verification after the fix (`winner_replay_audit_post_fix.json`):
- **Replay vs formal path:** equal for 109/109 pairs and all 810 launches: kernel sequence, grid,
  constexprs, warps, LDS and byte-identical code objects.
- **Scope of the change:** the replay changed for exactly the 8 pairs. The other 101 are
  byte-identical to before.
- **Default and autotune paths:** the old and new binaries of the 8 pairs have identical `.text` and
  launch metadata (`default_autotune_machine_code_check.txt`). Only DWARF line info differs, because
  source lines shifted.
- **Kernel manifest:** the re-probe of the three operators reproduced `kernel_counts.json` row for row
  (same counts, names and `first_call_identical`).
- **The 8 new reports** (`rerun_verification_2026-10-04.txt`):
  - capture validation passes;
  - Grid / Workgroup / LDS of every dispatch in all 13 passes equal the formal path;
  - the PC-sampling code objects have the formal-path sizes;
  - the reports re-analyze cleanly.
- **Local tree:** nothing else under `outputs/rocprof_compute/MI300X` changed.
- **HF remote audit** (`hf_remote_audit_2026-10-04.json`):
  - 6652 files before and after, 0 missing, 0 extra, 0 differing;
  - only the 8 pair directories plus sweep_log.json / coverage.json changed;
  - the NVIDIA_* and Neuron_trn2 folders are unchanged.
