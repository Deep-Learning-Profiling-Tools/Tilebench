# matmul_fp32_fp16_fp8 — why TileLang is slower on fp16 / fp8

Scope: fp16 and fp8_e4m3fn only. fp32 has a separate, already-established cause
(TF32 never reaches `tcgen05.mma` — see "fp32" at the end for the one-line
pointer).

## Verdict

At **matched, autotuned tile configs and with warp specialization enabled**,
TileLang is still last on both dtypes. The cause is **not** MMA serialization,
**not** a bad tile choice, **not** a warp-eligibility / latency-hiding failure,
and **not** a higher stall fraction.

All three kernels are latency-bound in the same way — 87-95% of active cycles
issue nothing, 94-98% of resident warps are stalled at any instant. What
separates them is **instruction count for identical MMA work**: TileLang emits
**2.11x (fp16) / 3.00x (fp8) more instructions than Triton**, concentrated in
the vector ALU/FMA/XU pipes, while emitting almost nothing on the **uniform
(scalar) datapath** that Triton and cuTile use heavily for loop and address
arithmetic. Its warp-specialized schedule recovers 1.60x / 2.00x of that through
a higher issue rate, and the residual — 1.33x / 1.50x — is the measured gap.

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
2.85x Triton's resident warps (11.33 vs 3.98 per cycle).

> **Corrected below.** It is tempting to read "more warps, less tensor
> throughput" as the extra warps wasting issue slots on bookkeeping. The
> scheduler data (see "Scheduler statistics") shows the opposite: those warps
> *earn their keep*, raising TileLang's issue rate 1.60x over Triton's. They are
> not the problem — they are a partial fix for it.

Two stalls are TileLang-exclusive:

| Stall (per issue-active) | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| `lg_throttle` | **1.088** | 0 | 0 |
| `branch_resolving` | **0.952** | 0.088 | 0.214 |
| `math_pipe_throttle` | **0.118** | 0 | 0.001 |

> **Read with care.** These are `per_issue_active` ratios, normalized per kernel
> — see "What the warps are actually stalled on" for why that form misleads.
> `math_pipe_throttle` at 0.118 against a `long_scoreboard` of 21.61 is noise,
> not evidence that the scalar pipes are backing up; that earlier reading was
> wrong. `branch_resolving` does survive normalization, and SASS locates it:
> `BSSY`/`BSYNC` at 395,520 against Triton's zero.

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

**This table is not comparable across backends as written, and the conclusion
originally drawn from it was wrong.** Absolute warp-cycles scale with how many
warps are resident. TileLang runs **2.83 active warps/scheduler against Triton's
0.99** — 2.85x. So most of the "7.0x more `long_scoreboard`" is a warp-count
artifact, not a longer wait.

Normalizing by resident warps (stall warps / active warps, i.e. *what fraction
of resident warp-time is stalled*) removes the artifact:

| | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| `long_scoreboard` | 72.7% | 39.5% | **81.2%** |
| total stalled | 97.0% | 94.2% | **98.1%** |

cuTile stalls a *larger* fraction of its warps on data than TileLang and is 27%
faster. Every backend here has 94-98% of its resident warps stalled at any
instant. **Stall fraction does not separate these kernels.** See the next
section for what does.

## Scheduler statistics: it is not an eligibility problem

`SchedulerStats` was missing from the original capture, so eligible-warp counts
were never collected. Recaptured (`reports/sch_*.ncu-rep`, same shape and
matched tiles, `harness/run_ncu_sched.sh`). Per scheduler, averaged over active
cycles. Raw metric dump: `analysis/metrics_sched_all.json` (1,896 metrics across
the six kernels, including every `issue_stalled` and `inst_executed_pipe_*`
counter used in this report).

| fp16 | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| Active warps / sched | **2.834** | 1.000 | 1.720 |
| Eligible warps / sched | **0.1144** | 0.0596 | 0.0483 |
| Issued warps / sched | **0.0951** | 0.0596 | 0.0480 |
| % cycles with **no** eligible warp | **90.5%** | 94.0% | 95.2% |

| fp8 | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| Active warps / sched | **2.745** | 1.017 | 1.694 |
| Eligible warps / sched | **0.1863** | 0.0666 | 0.0563 |
| Issued warps / sched | **0.1333** | 0.0666 | 0.0556 |
| % cycles with **no** eligible warp | **86.7%** | 93.3% | 94.4% |

The natural hypothesis — TileLang has 3x the warps yet cannot hide latency, so
something must be hurting warp *eligibility* — is **false**. TileLang has the
**most** eligible warps and the **fewest** starved cycles of the three. Its
warp-specialized schedule is doing exactly what it is supposed to do: the extra
warps buy a 1.60x (fp16) / 2.00x (fp8) higher issue rate than Triton.

All three kernels are deeply latency-bound: 87-95% of active cycles issue
nothing at all. In that regime runtime is not set by any pipe saturating; it is
set by **how many instructions must be pushed through at a near-fixed issue
rate**.

## The gap decomposes exactly into instruction count vs issue rate

Since `cycles = instructions / issue_rate`, the cycle ratio is forced:

| vs Triton | instructions/SM | issue rate | predicted cycles | measured cycles |
|---|---:|---:|---:|---:|
| fp16 | 39,399 / 18,710 = **2.11x** | **1.60x** | 2.11 / 1.60 = **1.32x** | **1.33x** |
| fp8 | 33,726 / 11,245 = **3.00x** | **2.00x** | 3.00 / 2.00 = **1.50x** | **1.50x** |

The identity closes to within measurement noise. TileLang issues 2.1x (fp16) /
3.0x (fp8) more instructions for **identical** MMA work (56,680 tensor-busy
cycles, all three), and its extra warps recover only 1.6-2.0x of that. The
residual is the gap.

Where the excess sits (% of peak sustained per pipe, so a *rate*; multiply by
the 1.33x cycle ratio for absolute counts):

**fp16**, matched tile, M=N=4096, K=2048:

| pipe | | TileLang | Triton | cuTile | TL/TR |
|---|---|---:|---:|---:|---:|
| `alu` | integer/logic, vector | 6.49% | 2.41% | 2.23% | 2.69x |
| `fma` | float multiply-add, vector | 2.71% | 0.33% | 0.15% | 8.15x |
| `xu` | transcendental / convert | 0.47% | 0.10% | 0.07% | 4.52x |
| `cbu` | branch / reconvergence | 0.80% | 0.02% | 0.07% | **43.19x** |
| **`uniform`** | **scalar datapath** | **0.23%** | **4.32%** | **3.04%** | **0.05x** |
| `lsu` | load/store | 0.72% | 1.29% | 0.85% | 0.56x |
| `adu` | address generation | 3.80% | 4.03% | 2.36% | 0.94x |
| `tma` | tensor memory accelerator | 0.27% | 0.61% | 0.50% | 0.44x |
| `tmem` | tensor memory | 0.01% | 0.04% | 0.15% | 0.18x |
| `tc` | tensor core issue | 0.48% | 0.63% | 0.69% | 0.75x |

**fp8_e4m3fn**, same shape:

| pipe | TileLang | Triton | cuTile | TL/TR |
|---|---:|---:|---:|---:|
| `alu` | 13.68% | 3.56% | 3.66% | 3.85x |
| `fma` | 3.25% | 0.57% | 0.24% | 5.67x |
| `xu` | 0.78% | 0.19% | 0.12% | 4.01x |
| `cbu` | 0.71% | 0.03% | 0.11% | **20.58x** |
| **`uniform`** | **0.16%** | **4.01%** | **2.75%** | **0.04x** |
| `lsu` | 0.67% | 0.67% | 0.85% | 0.99x |
| `adu` | 3.63% | 4.33% | 2.72% | 0.84x |
| `tma` | 0.13% | 0.61% | 0.32% | 0.22x |
| `tmem` | 0.01% | 0.07% | 0.28% | 0.16x |
| `tc` | 0.39% | 0.59% | 0.64% | 0.67x |

Note `lsu` (0.56x / 0.99x), `adu` (0.94x / 0.84x) and `tma` (0.44x / 0.22x) are
at or **below** Triton on both dtypes. TileLang is not issuing more loads, more
address-generation, or more TMA work — the excess is confined to the vector
arithmetic and branch pipes, which is what makes the `uniform` row meaningful
rather than incidental.

The `uniform` row is the most suggestive line in this report. Triton and cuTile
push loop counters, predicates and address arithmetic onto the **uniform
datapath** (uniform registers, one lane per warp). TileLang emits essentially
none — it does that work in the per-thread vector ALU instead, across all 384
threads. That is a concrete, named codegen difference and a plausible source of
a 2-3x instruction multiplier.

**Correction to an earlier retraction in this report.** The instruction-volume
hypothesis was previously dismissed on the grounds that FMA sits at 2.7% of
peak, and nothing at 2.7% utilization can be a throughput bottleneck. That test
was the wrong one: it rules out *pipe saturation*, which was never the claim.
In a kernel that issues nothing 90% of cycles, instruction count still sets
runtime through issue-slot and dependency-chain length. The low utilizations
are consistent with instruction count mattering, not evidence against it.

## SASS confirmation

`--set source --section SourceCounters` gives per-instruction **dynamic** counts
(each SASS line weighted by its own `Instructions Executed`). fp16, matched
tile, M=N=2048, K=4096, `harness/sass_opcodes.py`:

| | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| uniform-datapath opcodes (`U*`) | **13.90%** | **55.56%** | 36.69% |
| instructions with any `UR*` operand | 39.98% | 70.09% | 66.49% |

This confirms the `uniform`-pipe reading directly, and more starkly than the NCU
counter did. Named opcodes, dynamic counts:

| opcode | what it is | TileLang | Triton | cuTile |
|---|---|---:|---:|---:|
| `R2UR` | move vector reg -> uniform reg | **397,568** | **1,024** | 11,008 |
| `IMAD` | integer multiply-add, vector datapath | **2,350,408** | 60,416 | 206,578 |
| `BSSY`/`BSYNC` | divergent-branch sync stack | **395,520** each | **0** | ~2,500 |
| `NANOSLEEP` | spin-wait backoff | **1,144,101** | **0** | 458,259 |
| `PRMT`+`UPRMT` | byte permute (swizzle/layout) | **909,312** | **0** | **0** |
| `SYNCS` | warp sync | 2,765,642 | 52,224 | 1,014,566 |
| `UTCHMMA` | the actual MMA | **65,536** | **32,768** | 65,536 |

Reading these:

- **`R2UR` at 388x Triton** is the mechanism behind the `uniform` gap. TileLang
  does not fail to use uniform registers — it computes the values in the *vector*
  datapath first and then copies them across. That is the expensive way to get a
  scalar: every thread computes it, then one result is kept.
- **`IMAD` at 39x Triton** is per-thread integer address arithmetic — the work
  Triton keeps on `UIADD3`/`ULEA`.
- **`BSSY`/`BSYNC` at ~0 for Triton** means Triton's inner loop has *no*
  divergent control flow. TileLang's warp-specialized producer/consumer code is
  full of it.
- **`PRMT`/`UPRMT` only in TileLang** — 909K byte-permute ops that neither
  competitor emits at all, i.e. swizzle/layout computed at runtime rather than
  folded into descriptors.
- **`UTCHMMA` 65,536 vs Triton's 32,768.** TileLang issues **2x the MMA
  instructions** for the same math — a smaller `tcgen05` atom. Tensor-busy
  *cycles* are identical (56,680), so this is not extra work; it is the same work
  in twice as many instructions, with twice the surrounding bookkeeping.

Full opcode table: `analysis/sass_opcodes_fp16.txt`.

### Memory path: all three load via TMA, only TileLang stores without it

Complete memory-opcode inventory (`harness/sass_memops.py`,
`analysis/sass_memops_fp16.txt`) — `sass_opcodes.py` prints only a top-N union
and hides low-count-but-decisive opcodes:

| opcode | | TileLang | Triton | cuTile |
|---|---|---:|---:|---:|
| `UTMALDG` | TMA bulk tensor load | 49,152 | 33,792 | 49,152 |
| `LDG` / `LDGSTS` | plain / `cp.async` global load | **0** | **0** | **0** |
| `UTMASTG` | TMA bulk tensor store | **0** | 256 | 512 |
| `STG` | plain global store | **8,192** | **0** | **0** |

**Loads: yes, uniformly.** Every backend feeds operands exclusively through TMA.
No one falls back to a plain or `cp.async` path, so the load mechanism is not a
differentiator.

**Stores: no.** TileLang is the only backend that writes its output with plain
global stores; Triton and cuTile use a TMA store. This is independent
corroboration of the K-sweep, which put the prologue/epilogue at 14-16% of the
gap with TileLang's intercept 2.05x worse — two unrelated methods landing on the
same component.

Three further rows, reported as observations only:

| opcode | | TileLang | Triton | cuTile |
|---|---|---:|---:|---:|
| `MEMBAR` | memory barrier | **16,384** | 256 | 2,048 |
| `FENCE` | fence | **16,640** | 768 | 2,304 |
| `YIELD` | warp yield | **65,536** | **0** | **0** |
| `LDS` / `STS` | shared load / store | 19,456 / 256 | 448 / 16,512 | 2,048 / 16,896 |
| `LDL` / `STL` | local (register spill) | 2,048 / 3,072 | 9,472 / 9,472 | 0 / 0 |

- **64x the `MEMBAR` and 21x the `FENCE` of Triton.** Fences serialize memory
  operations; at ~16K executions this is unlikely to be free, and it appears in
  no NCU counter examined in this report.
- **`YIELD` 65,536 against zero for both competitors**, consistent with the
  `NANOSLEEP` spin-wait pattern.
- **`LDS` 43x Triton while `STS` runs 64x the other way.** TileLang reads shared
  memory far more and writes it far less, suggesting operands are staged through
  registers where the competitors feed the tensor core from shared directly.
- Note Triton spills *more* than TileLang (`LDL`/`STL` 9,472 vs 2,048/3,072) and
  is still faster — spill volume is not the discriminator here.

None of these five rows is tied to time. They are located, not costed.

**Caveat:** this capture is at a different shape (M=N=2048, K=4096) than the
counter run (M=N=4096, K=2048), and the source page sums over the whole grid
rather than averaging per SM. So the *total* instruction ratios here are not
comparable to the 2.11x from the counter table — only the **composition** and
the **per-opcode ratios** should be read. fp16 only; fp8 not captured.

## What is still not established

**That the excess instructions are causal rather than correlated.** The
`cycles = instructions / issue_rate` identity is exact, but it is an identity —
it relocates the question rather than answering it. The missing experiment is
an intervention: reduce TileLang's emitted instruction count at a fixed tile and
show cycles fall proportionally. Nothing here does that.

The SASS above names the code patterns but still does not time them. Remaining:

1. **fp8 SASS**, to check the same patterns hold where the gap is larger (1.50x).
2. **`T.tcgen05_gemm` async variant**, to separate "inherent to warp-specialized
   lowering" from "`T.gemm`'s in-loop `mbarrier_wait_parity`."
3. **MMA atom width.** TileLang emits 2x the `UTCHMMA` of Triton for identical
   tensor cycles. Whether forcing the wider atom is possible from the TileLang
   API — and what it is worth — is untested, and is the most directly
   actionable item here.

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
| instructions / SM | **39,399** | **18,710 (2.11x fewer)** |

Same tile, same depth, same shared memory, same MMA cycles — and 2.11x the
instructions. **The difference is therefore in the code TileLang generates for
that pipeline, not in the schedule it was asked to generate.** This is a
lowering quality issue, and no stage sweep is needed to establish it.

The shape of the difference:

| | TileLang | Triton |
|---|---:|---:|
| threads / warps per CTA | 384 / 12 | 128 / 4 |
| registers per thread | 168 | 255 |
| active warps / scheduler | 2.83 | 1.00 |
| issue rate / scheduler | 0.0951 | 0.0596 |
| instructions / SM | 39,399 | 18,710 |
| `uniform` pipe (% peak) | 0.23% | 4.32% |

Triton drives the pipeline with 4 warps that block on real barriers and keeps
scalar/address arithmetic on the uniform datapath. TileLang emits a 12-warp
warp-specialized producer/consumer schedule; the extra warps successfully raise
its issue rate 1.60x, but it has 2.11x the instructions to issue, so it still
loses 1.33x on cycles.

Note that the `barrier`-vs-`long_scoreboard` accounting difference between the
two (mbarrier spins land in `long_scoreboard`) is real but, per the normalized
table above, does not by itself separate them — cuTile has a *higher*
`long_scoreboard` fraction than TileLang and is faster.

## Reproducing

```bash
source harness/env.sh
PREFIX=f_ bash harness/run_ncu_matched.sh          # 8 profiles at tuned tiles
bash harness/run_ncu_sched.sh                     # SchedulerStats: eligible warps
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python:$PYTHONPATH \
  $PY ~/.claude/skills/ncu-report-skill/helpers/analyze_reports.py --run-dir . \
    --report reports/f_tilelang_ws_fp16.ncu-rep --tag tl_fp16 \
    --report reports/f_triton_fp16.ncu-rep      --tag tr_fp16 \
    --report reports/f_cutile_fp16.ncu-rep      --tag ct_fp16

# scheduler budget: eligible warps + cycles = instructions / issue_rate
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python:$PYTHONPATH \
  $PY ~/.claude/skills/kernel-perf-differential/helpers/scheduler_budget.py \
    reports/sch_tilelang_ws_fp16.ncu-rep+reports/m_tilelang_ws_fp16.ncu-rep:tilelang \
    reports/sch_cutile_fp16.ncu-rep+reports/m_cutile_fp16.ncu-rep:cutile \
    reports/sch_triton_fp16.ncu-rep+reports/m_triton_fp16.ncu-rep:triton

# SASS: per-instruction dynamic opcode histogram (M=N=2048, K=4096)
$NCU --profile-from-start off --set source --section SourceCounters \
     --target-processes all --force-overwrite -o reports/src_triton_fp16 \
     $PY -u harness/profile_matmul.py --backend triton --dtype fp16 \
        --M 2048 --N 2048 --K 4096 --matched
$PY harness/sass_opcodes.py \
    reports/src_tilelang_ws_fp16.ncu-rep:tilelang \
    reports/src_triton_fp16.ncu-rep:triton \
    reports/src_cutile_fp16.ncu-rep:cutile
# full memory / TMA / fence inventory (catches UTMASTG, STG, MEMBAR)
$PY harness/sass_memops.py \
    reports/src_tilelang_ws_fp16.ncu-rep:tilelang \
    reports/src_triton_fp16.ncu-rep:triton \
    reports/src_cutile_fp16.ncu-rep:cutile

$PY harness/final_headtohead.py      # tuned-vs-tuned timings, all dtypes
$PY harness/ws_flag_check.py         # warp-spec flag A/B on benchmark inputs
$PY harness/fp8_localize.py          # fp8 bn=256 tile sweep
$PY harness/did_it_pass.py           # broken fp8 vs the real verifier
$PY harness/fp8_atom.py              # which tcgen05 atom each case selects
```
