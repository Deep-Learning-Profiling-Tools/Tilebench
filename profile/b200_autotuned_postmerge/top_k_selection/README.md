# Top-K Selection: PTX / SASS / NCU run, 2026-07-26

Follow-up to `profile/top_k_selection_ncu_20260720`. That run established that
TileLang's compare-exchange kernel executes several times Triton's instructions.
This run identifies the causes and backs each with NCU / PTX / SASS evidence,
plus an attribution probe that measures how much each cause is worth.

Read `REPORT.md`. It is a diagnosis, not an optimization plan — the probe
variants in `harness/variants_tilelang.py` are diagnostic instruments and
deliberately break source parity with the other backends, so they are not
proposed changes to the operator.

- Backends: TileLang, Triton, cuTile, at the autotune winners recorded in
  `results/b200_autotuned_postmerge/operators/top_k_selection.json`.
- Shapes: low N = 4,096; high N = 1,048,576. Pass `stage = N/64`, `stride = 16`.
- Hardware: NVIDIA B200 SM 10.0, NCU 2026.1.1, pinned by GPU UUID to the
  SLURM-allocated card.

## Layout

| Path | Contents |
|---|---|
| `harness/env.sh` | env python, `PYTHONPATH`, `NVCC_PREPEND_FLAGS`, GPU pin |
| `harness/profile_topk.py` | single-launch harness for the three backends |
| `harness/dump_ir.py` | PTX / SASS / cubin / TileIR extraction |
| `harness/variants_tilelang.py` | four TileLang probe variants for attribution |
| `harness/profile_variant.py` | probe launcher, with bit-exact verification |
| `harness/run_ncu.sh` | the 12 backend NCU collections |
| `harness/analyze_ir.py` | static PTX / SASS instruction mixes |
| `harness/analyze_ncu.py` | key metrics, stalls, dynamic opcode counts |
| `harness/analyze_variants.py` | variant vs reference comparison table |
| `ir/` | PTX, SASS, cubins, generated CUDA, TileIR bytecode |
| `ir/variants/` | same for the four TileLang probe variants |
| `reports/` | 12 backend + 4 probe `.ncu-rep` |
| `analysis/` | generated metric tables and SASS streams |
| `logs/` | harness stdout under NCU |

cuTile has no PTX stage — TileIR lowers straight to cubin, so only SASS and
TileIR bytecode are available for it.
