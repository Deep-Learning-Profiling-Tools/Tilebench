# matmul_fp32_fp16_fp8 — why TileLang is slower on fp16 / fp8

Scope: fp16 and fp8_e4m3fn only. fp32 has a separate, already-established cause
(TF32 never reaches `tcgen05.mma` — see "fp32" at the end for the one-line
pointer).

## Verdict

At **matched, autotuned tile configs and with warp specialization enabled**,
TileLang is still last on both dtypes. The cause is **not** MMA serialization
and **not** a bad tile choice. It is that TileLang issues **~2x the instructions
per active cycle for identical MMA work**, and that excess sits almost entirely
in the scalar pipes (FMA, XU, ALU).

CUDA-graph timed, M=N=4096, K=8192, each backend at its own recorded autotune
winner, all passing `verify(atol=5.0, rtol=0.1)`:

| dtype | TileLang | Triton | cuTile | torch |
|---|---:|---:|---:|---:|
| fp16 | 1109 | 1271 (1.15x) | 1403 (1.27x) | 1355 |
| fp8_e4m3fn | 2077 | 2838 (1.37x) | 3093 (1.49x) | 2453 |

The autotuner is **not** at fault: TileLang's tuned winner is the *same tile* as
Triton's (fp16 256x256x64, fp8 256x256x128).

## Setup

- B200, sm_100, 148 SMs, pinned by UUID. NCU 2026.1.1.
- **NCU**: M=N=4096, K=2048, every backend forced to the tuned tile
  (`--matched`). 4096 gives 256 CTAs so all three see the same 1.73 waves/SM.
- **Timings**: CUDA-graph replays at M=N=4096, K=8192, on the benchmark's own
  `1/sqrt(K)`-scaled inputs.
- `tilelang_ws` = warp specialization on. This is now what
  `impl_tilelang.py` ships (`TL_DISABLE_WARP_SPECIALIZED: False`).
- Collected with `--section SpeedOfLight --section ComputeWorkloadAnalysis
  --section WarpStateStats --section Occupancy --section LaunchStats`.
  `MemoryWorkloadAnalysis`, `InstructionStats`, and
  `sass__inst_executed_per_opcode` each push the replay-pass count into a hang
  at this shape and were dropped.

## The evidence

fp16, matched 256x256x64, M=N=4096 K=2048
(`analysis/compare_tl_fp16_vs_tr_fp16_vs_ct_fp16.txt`):

| Metric | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| `gpu__time_duration.sum` (us) | 76,960 | 61,856 | 57,824 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | **53.95** | 71.86 | 77.82 |
| `sm__inst_executed.avg.per_cycle_active` | **0.375** | 0.237 | 0.190 |
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | **2.709** | 0.333 | 0.154 |
| `sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active` | **6.491** | 2.420 | 2.219 |
| `sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active` | **0.474** | 0.105 | 0.066 |
| `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active` | 0.718 | 1.296 | 0.848 |
| `l1tex__throughput.avg.pct_of_peak_sustained_active` | 40.46 | 53.90 | 58.36 |
| `launch__block_size` | 384 | 128 | 256 |
| `sm__warps_active.avg.per_cycle_active` | 11.33 | 3.98 | 6.79 |
| `launch__registers_per_thread` | 168 | 255 | 255 |

fp8_e4m3fn, matched 256x256x128
(`analysis/compare_tl_fp8_vs_tr_fp8_vs_ct_fp8.txt`):

| Metric | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| `gpu__time_duration.sum` (us) | 46,560 | 34,304 | 32,704 |
| `sm__pipe_tensor_cycles_active...active` | **44.18** | 66.07 | 70.09 |
| `sm__inst_executed.avg.per_cycle_active` | **0.526** | 0.262 | 0.219 |
| `...pipe_fma...active` | **3.251** | 0.573 | 0.241 |
| `...pipe_alu...active` | **13.69** | 3.555 | 3.613 |
| `...pipe_xu...active` | **0.777** | 0.194 | 0.120 |
| `l1tex__throughput...active` | 33.13 | 49.55 | 52.57 |

### Reading it

All three run the identical tile, the identical grid (256 CTAs), and identical
waves/SM (1.73), so total MMA work is identical. `sm__pipe_tensor_cycles_active`
is a *ratio* — tensor-busy cycles over active cycles. Same numerator, so the
denominator is what differs: TileLang simply has more active cycles, filled with
non-MMA instructions.

Per active cycle TileLang issues **1.97x** cuTile's instruction count on fp16
(0.375 vs 0.190) and **2.40x** on fp8 (0.526 vs 0.219). The excess is scalar:

| Pipe | fp16 TL/cuTile | fp8 TL/cuTile |
|---|---:|---:|
| FMA | **17.6x** | **13.5x** |
| XU | **7.2x** | **6.5x** |
| ALU | **2.9x** | **3.8x** |
| LSU | 0.85x | 0.79x |

LSU is *lower* — TileLang is not doing more loads. It is doing more arithmetic
and more conversions around the same data movement. `l1tex__throughput` being
lower (40.5 vs 58.4) is a consequence of the longer runtime, not a cause: the
same bytes spread over more cycles.

TileLang also runs 384 threads/CTA against cuTile's 256 and Triton's 128, with
2.85x Triton's resident warps (11.33 vs 3.98 per cycle). More warps, less tensor
throughput — the extra warps are consuming issue slots on bookkeeping.

Two stalls are TileLang-exclusive and corroborate scalar pressure:

| Stall (per issue-active) | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| `lg_throttle` | **1.088** | 0 | 0 |
| `branch_resolving` | **0.952** | 0.088 | 0.214 |
| `math_pipe_throttle` | **0.118** | 0 | 0.001 |

`math_pipe_throttle` being nonzero only for TileLang means its scalar pipes are
actually beginning to back up — the only kernel of the three where that happens.

## Retracted: "synchronous `T.gemm` serializes the MMA"

An earlier revision of this report blamed `T.gemm` being synchronous on
Blackwell — it inserts `mbarrier_wait_parity` right after issuing the MMA group,
so two groups are never in flight. The generated CUDA does show that handshake,
and it is documented in `T.gemm`'s own docstring.

**Cycle attribution does not support it as the limiter:**

| `barrier` stall (per issue-active) | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| fp16 | **0.177** | 5.133 | 0.198 |
| fp8 | **0.210** | 4.335 | 0.264 |

TileLang has the *lowest* barrier stall of the three on both dtypes. Triton
carries **29x** TileLang's barrier stall on fp16 and is still 24% faster. If the
handshake were the limiter, this ordering would be reversed.

The supporting BK 64->128 experiment (+33%, 711 -> 946 TFLOP/s) was run at
128x128 on the *default* config with warp specialization off, and does not
transfer to the tuned point. At the tuned 256x256, BK=128 does not even build:

```
256x256x64  st3   64 KB/stage   builds
256x256x128 st3  128 KB/stage   FAILED: Failed to set the allowed dynamic shared mem
256x256x128 st2  128 KB/stage   FAILED: Failed to set the allowed dynamic shared mem
```

128 KB/stage x 3 stages exceeds the 227 KB SMEM budget. That is why the
autotuner picked BK=64 at 256x256 — the lever is unavailable, not unexplored.

## What the warp-spec flag was worth

`impl_tilelang.py` shipped `TL_DISABLE_WARP_SPECIALIZED: True` hardcoded.
Flipping it to `False` (now applied) at the tuned tile, benchmark inputs:

| | ws=off (was shipped) | ws=on (now) | gain |
|---|---:|---:|---:|
| fp16 | 953 TFLOP/s, maxabs 0.0057 | **1100**, maxabs 0.0000 | +15% |
| fp8 | 1790 TFLOP/s, maxabs 0.0801 | **2026**, maxabs 0.0801 | +13% |

NCU corroborates on fp8 (the one dtype where both were captured):
duration 50,816 -> 46,560 us, tensor pipe 43.17% -> 46.91%, barrier stall
2.33 -> 0.21.

This is a speed fix. It also improves fp16 accuracy slightly (maxabs 0.0057 ->
0.0000, bit-matching torch), but both configurations pass the suite.

## Correctness: the fp8 `.ws` atom bug (does not affect the benchmark)

`tcgen05.mma.ws.kind::f8f6f4` at `atom_n=256` miscomputes the upper half of
every N tile. Reproduced on unscaled `randn` inputs, M=N=512, K=1024:

```
fp8 bn=256: cols   0-127 max err 0.0000   cols 128-255 max err 208.0000
fp16 control at bn=256: rel err 3.59e-04   -> OK, so it is fp8-specific
```

Tile sweep — wrong **iff** `BLOCK_SIZE_N == 256`, independent of BM, BK, threads:

```
128x128x64  OK     128x256x64  WRONG
256x128x64  OK     256x256x64  WRONG
128x128x128 OK     256x256x128 WRONG
```

Mechanism: `GetTCGEN5MMAMeta` (`src/op/tcgen5_meta.h`) tries
`ws_valid_atom_ns = {256, 128, 64}` *before* the non-ws branch for fp8, so any
fp8 GEMM with `N % 256 == 0` lands on `tcgen05mma_ws_ss` at `atom_n=256`. The
fp16 branch has no ws path at `M % 128 == 0`, which is why fp16 is immune.
Confirmed in the generated CUDA: fp8 emits `tcgen05mma_ws_ss<kFloat8_e4m3>`,
fp16 emits `tcgen05mma_ss<kFloat16, false>` (`ir/atom_*.cu`).

**This does not affect TileBench results.** `generate_matmul_fp32_fp16_fp8_inputs`
scales operands by `1/sqrt(K)`, so outputs are O(1) and the defect stays far
inside tolerance:

```
config.yaml tolerances: atol=5.0  rtol=0.1
K=1024   BROKEN bn=256   verify -> PASS   maxabs 0.25
K=8192   BROKEN bn=256   verify -> PASS   maxabs 0.08
K=20480  BROKEN bn=256   verify -> PASS   maxabs 0.05
```

`tilelang_ok: true` in the recorded results is a correct verdict, and fp8
timings are valid. It is an upstream TileLang bug worth reporting, not a
benchmark-side problem. It does leave TileLang's fp8 ~20x less accurate than
Triton/cuTile (maxabs 0.055-0.164 vs 0.004-0.008) even where it passes.

No workaround is applied: constraining fp8 to `BLOCK_SIZE_N<=128` costs real
performance (1597 vs 2023 TFLOP/s) to fix an error the benchmark never sees.

## fp32

Separate, already-established cause: `use_tmem = dtype != "float32"` forces the
register path because `GetTCGEN5MMAMeta` has no TF32 operand branch — TF32 never
reaches `tcgen05.mma`. Not re-analyzed here.

## Localizing the gap: K-loop vs epilogue

Counters correlate; they do not establish cause. Since NCU's utilization
metrics all carry runtime in the denominator, a slower kernel shows lower
utilization *everywhere* — the symptom is indistinguishable from the cause.

A controlled decomposition avoids that. Tile and grid are fixed, so sweeping K
varies only the K-loop trip count. Fit `time = intercept + slope * K`:
intercept is prologue + epilogue (O(1)), slope is steady-state per-K cost.
From the tuned graph-timed runs at K = 2048 / 8192 / 20480 (`/tmp/kfit.py`
arithmetic over `logs/final_headtohead.log`):

| fp16 | intercept (ms) | slope (ms per 1k K) | R^2 |
|---|---:|---:|---:|
| TileLang | 0.01459 | **0.02861** | 0.99999 |
| Triton | 0.00740 | 0.02520 | 0.99992 |
| cuTile | 0.00710 | 0.02305 | 1.00000 |

| Gap at K=8192 | total | from K-loop | from prologue/epilogue |
|---|---:|---:|---:|
| fp16 vs cuTile | 0.0530 ms | **0.0455 (86%)** | 0.0075 (14%) |
| fp16 vs Triton | 0.0351 ms | **0.0280 (80%)** | 0.0072 (20%) |
| fp8 vs cuTile | 0.0421 ms | **0.0352 (84%)** | 0.0069 (16%) |
| fp8 vs Triton | 0.0318 ms | **0.0205 (65%)** | 0.0112 (35%) |

R^2 >= 0.9993 on every fit, so the linear model is sound.

**Conclusion: 80-86% of the gap is steady-state K-loop cost.** TileLang pays
**1.24x (fp16) / 1.44x (fp8)** more per K-tile than cuTile.

**This rules out the epilogue as the primary cause.** TileLang's intercept is
2.05x cuTile's, so the tmem->register->global epilogue *is* relatively worse,
but it contributes only 14-16% of the gap in absolute terms. The 17.6x FMA
ratio cannot be primarily epilogue-driven.

Combined with the tensor-cycle identity above (56,680 cycles for all three),
the picture is: identical MMA work, but each K iteration takes ~25-45% longer
to traverse.

## What the warps are actually stalled on

The `smsp__average_warps_issue_stalled_*_per_issue_active.ratio` family is
normalized by each kernel's *own* issue-active cycles, which differ ~2.9x here.
Read raw, it says cuTile stalls more on `long_scoreboard` than TileLang
(28.81 vs 21.61) — an artifact of the denominator. Converting to absolute
warp-cycles (`ratio x issue_active.avg.per_cycle_active x sm__cycles_active.avg`)
reverses the conclusion.

fp16, matched tuned tile:

| Stall | Meaning | TileLang | Triton | cuTile |
|---|---|---:|---:|---:|
| `long_scoreboard` | **waiting on global/L1 data** | **216,122 (75.0%)** | 31,011 (41.9%) | 100,569 (82.8%) |
| `short_scoreboard` | MIO: smem + address gen | 18,899 (6.6%) | 6,406 (8.7%) | 8,397 (6.9%) |
| `wait` | **arithmetic dependency** | 18,522 (6.4%) | 5,954 (8.0%) | 4,987 (4.1%) |
| `lg_throttle` | LSU queue full | 10,879 (3.8%) | 0 | 0 |
| `branch_resolving` | branch | 9,516 (3.3%) | 412 (0.6%) | 745 (0.6%) |
| `barrier` | sync | 1,774 (0.6%) | 24,128 (32.6%) | 693 (0.6%) |
| **TOTAL stall warp-cycles** | | **288,214 (2.37x)** | 74,046 (0.61x) | 121,422 (1.00x) |

fp8 is the same shape: TileLang 167,031 total (2.52x cuTile),
`long_scoreboard` 76.5%.

**Conclusion: the limiter is memory-latency exposure in the K-loop, not
arithmetic.** 75% of TileLang's stall time is warps waiting for tile data, and
in absolute warp-cycles that is **2.15x cuTile's**. Per-instruction attribution
(`analysis/src_tilelang_ws.csv`) places those `long_scoreboard` hits on the
backward branches of mbarrier spin-wait loops, i.e. waiting for TMA arrival.

Arithmetic dependency (6.4%), address generation / MIO (6.6%) and branch
resolution (3.3%) together account for ~16% of the stall budget. They are
elevated relative to the others — `wait` is 3.7x cuTile, `branch_resolving`
12.8x — but they are not the mechanism.

This agrees with the K-sweep from an independent method: the gap is 86%
steady-state K-loop, and the K-loop's dominant cost is waiting on the next tile.

## What is still not established

*Which* property of the generated pipeline causes the longer wait.

What is ruled out: pipeline depth, tile shape, and SMEM budget. TileLang and
Triton are matched on all three **by construction and in the built binary**:

| | TileLang | Triton |
|---|---:|---:|
| tuned tile | 256x256x64 | 256x256x64 |
| `num_stages` | 3 | 3 |
| smem static + dynamic | 1,024 + 196,608 B | 0 + 196,656 B |
| A+B per stage x 3 | 196,608 B exactly | 196,608 B + 48 scratch |
| tensor-busy cycles | 56,680 | 56,680 |
| `long_scoreboard` warp-cyc | **216,122** | **31,011 (7.0x fewer)** |

Same tile, same depth, same shared memory, same MMA cycles — and 7x the data
wait. **The difference is therefore in the code TileLang generates for that
pipeline, not in the schedule it was asked to generate.** This is a lowering
quality issue, and no stage sweep is needed to establish it.

The shape of the difference:

| | TileLang | Triton |
|---|---:|---:|
| threads / warps per CTA | 384 / 12 | 128 / 4 |
| registers per thread | 168 | 255 |
| `long_scoreboard` | 216,122 (75.0%) | 31,011 (41.9%) |
| `barrier` | 1,774 (0.6%) | 24,128 (32.6%) |

Triton drives the pipeline with 4 warps that block on real barriers. TileLang
emits a 12-warp warp-specialized producer/consumer schedule whose consumers
spin on mbarriers — which is accounted as `long_scoreboard`, not `barrier` —
and nets 7x the data wait despite having 3x the warps available to hide it.

What remains untested is whether that is inherent to TileLang's
warp-specialized lowering or specific to `T.gemm`'s in-loop
`mbarrier_wait_parity`. A `T.tcgen05_gemm` async variant with the wait hoisted
out of the K-loop would separate them; it was not run.

Note also that instruction *volume* is unlikely to bind on its own: absolute
pipe utilizations are low (FMA 2.7%, XU 0.5%, ALU 6.5% of peak).

## Reproducing

```bash
source harness/env.sh
PREFIX=f_ bash harness/run_ncu_matched.sh          # 8 profiles at tuned tiles
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python:$PYTHONPATH \
  $PY ~/.claude/skills/ncu-report-skill/helpers/analyze_reports.py --run-dir . \
    --report reports/f_tilelang_ws_fp16.ncu-rep --tag tl_fp16 \
    --report reports/f_triton_fp16.ncu-rep      --tag tr_fp16 \
    --report reports/f_cutile_fp16.ncu-rep      --tag ct_fp16

$PY harness/final_headtohead.py      # tuned-vs-tuned timings, all dtypes
$PY harness/ws_flag_check.py         # warp-spec flag A/B on benchmark inputs
$PY harness/fp8_localize.py          # fp8 bn=256 tile sweep
$PY harness/did_it_pass.py           # broken fp8 vs the real verifier
$PY harness/fp8_atom.py              # which tcgen05 atom each case selects
```
