# Top-K Selection: causes of the TileLang instruction excess, with NCU / PTX / SASS evidence

## Verdict

All three backends express the compare-exchange step with the *same* source
arithmetic — `(offset // stride) * (2 * stride) + (offset % stride)` and
`((slice_1_offset // stage) % 2) == 1` appear verbatim in `impl_tilelang.py`,
`impl_triton.py` and `impl_cutile.py`. Nothing diverges until after the
frontend. **The gap is compiler codegen on identical input arithmetic.**

Five causes, ranked by measured contribution to TileLang's 3.37 M instruction
excess over Triton at high N:

| # | Cause | Evidence | Contribution |
|---|---|---|---|
| 1 | Index expression re-materialized per syntactic use (no CSE at TIR → C) | 74 `rmod` + 40 `rdiv` locals in generated CUDA for a 2-element body | — |
| 2 | Same tree re-emitted in int64 for addressing → runtime 32/64-bit division dispatch, 4 `CALL`s | 12 int64 division sites; `div.u32` + `div.s64` fast/slow paths in PTX | — |
| 3 | Reciprocal CSE fails: 11 `MUFU.RCP` vs Triton's 2 for the same 2 divisors | SASS; XU active 29.0% → 0.0% when division is removed | **1–3 together: 91% of instructions, 87% of time** |
| 4 | Per-access guard branches instead of predication | 24 static `BRA`, 9 `BSSY`/`BSYNC` pairs vs Triton's 1 and 0 | 5.5% of instructions, **0% of time** |
| 5 | Weaker vectorization: 2x the memory instructions for identical traffic | 32,768 vs 16,384 L1 requests at identical 131,072 sectors | residual |

Causes 1–3 are one mechanism seen at three stages of lowering: TileLang
re-materializes the division tree, emits it twice at two widths, and the
resulting basic-block fragmentation prevents ptxas from merging the reciprocals.
Together they account for **3.05 M of the 3.37 M excess instructions (91%)** and
**3.14 us of the 3.62 us duration gap (87%)** — measured by the attribution
probe below.

## Setup

- GPU: NVIDIA B200, SM 10.0, pinned by UUID to the SLURM-allocated card
  (job 9073602, IDX:0). NCU 2026.1.1. CUDA 13.2 / torch 2.12+cu130.
- Shapes: low N = 4,096; high N = 1,048,576. `k` is unused by the
  compare-exchange kernel.
- Profiled pass: `stage = N/64`, `stride = 16` — a real pass in all three
  pipelines.
- **Configs: the autotune winners recorded in
  `results/b200_autotuned_postmerge/operators/top_k_selection.json`**, which are
  identical for every `k` at a given N:

  | Backend | N = 4,096 | N = 1,048,576 |
  |---|---|---|
  | TileLang | `BLOCK_SIZE=512, threads=256` | same |
  | Triton | `BLOCK_SIZE=512, num_warps=8, num_stages=1` | same |
  | cuTile | `tile=1024, occupancy=8` | `tile=512, occupancy=16` |

- One kernel launch captured per NCU invocation, after three warmups.
- The full operator launches 78 such kernels at low N and 210 at high N. These
  are single-kernel profiles, not end-to-end Top-K timings.
- Run-to-run variance on these 6–10 us kernels is roughly ±4%. All durations
  quoted here come from one contemporaneous collection session.

Artifacts: `ir/` (PTX, SASS, cubins, generated CUDA, TileIR bytecode),
`reports/` (12 backend + 4 probe `.ncu-rep`), `analysis/` (metric tables),
`harness/` (all scripts).

> Note on the earlier run: `profile/top_k_selection_ncu_20260720` profiled
> Triton at `num_warps=4` (the 20260720 autotune winner). Under the postmerge
> winner, `num_warps=8`, Triton's thread block is 256 rather than 128, so each
> thread handles 2 elements instead of 4. That changes Triton's numbers
> materially — see the two flagged rows below — but changes nothing about the
> TileLang mechanism, whose config is identical in both runs.

---

## NCU evidence, high N

| Metric | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| Duration | **9.472 us** | **5.856 us** | 6.272 us |
| Executed warp instructions | **4.317 M** | 0.950 M | 1.397 M |
| IPC (issued, active) | 2.580 | 1.849 | 1.920 |
| SM throughput | **38.5%** | 15.3% | 19.3% |
| Memory throughput | 5.9% | **11.2%** | 9.3% |
| ALU active | **54.6%** | 39.0% | 45.6% |
| XU active | **29.4%** | 19.1% | 7.9% |
| Achieved occupancy | 62.6% | 67.8% | 37.0% |
| Registers/thread | 32 | 22 | 32 |
| Global-load L1 requests | **32,768** | **16,384** | 32,768 |
| Global-load L1 sectors | 131,072 | 131,072 | 131,072 |
| DRAM bytes read | 4.21 MB | 4.20 MB | 4.20 MB |
| Math-pipe throttle (warps/issue-active) | 2.610 | **2.713** | 1.601 |
| Wait stall | **3.151** | 2.229 | 1.701 |
| Long-scoreboard stall | 1.524 | **5.567** | 1.485 |
| Not-selected stall | 3.591 | 3.848 | 1.650 |

All three move identical DRAM traffic, so no backend is doing less work — only
more instructions per unit of it. TileLang's high SM throughput is not useful
computation: memory throughput is the lowest of the three while ALU activity and
dependency stalls are the highest. Triton is the most memory-facing version
(11.2% memory throughput, long-scoreboard-dominated at 5.567).

Two rows are worth flagging because they invert the 20260720 conclusion under
this config, and neither weakens the diagnosis:

- **Math-pipe throttle is now marginally higher for Triton (2.713) than
  TileLang (2.610).** At `num_warps=8` each Triton thread owns 2 elements
  instead of 4, so its single reciprocal amortizes over half as many divisions.
  Triton is *more* math-throttled per issue cycle while still executing 4.5x
  fewer instructions — the throttle ratio measures pipe pressure per issue, not
  total work.
- **XU active is 19.1% for Triton**, up from 9.5% at `num_warps=4`, for the same
  reason. TileLang's 29.4% is still the highest and is still entirely division
  (see cause 3).

Dynamic executed warp instructions by opcode (full table in
`analysis/ncu_metrics.txt`):

| Opcode | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| IMAD | 1,359,872 | 311,296 | 466,944 |
| ISETP | 851,968 | 147,456 | 286,720 |
| VIADD | 237,568 | 73,728 | 106,496 |
| SEL | 212,992 | 65,536 | 90,112 |
| LOP3 | 212,992 | 65,536 | 102,400 |
| IADD3 | 204,800 | 24,576 | 0 |
| BRA | **188,416** | **0** | **0** |
| BSSY + BSYNC | **147,456** | **0** | **0** |
| MUFU | 81,920 | 16,384 | 8,192 |
| I2F | 81,920 | 16,384 | 8,192 |
| F2I | 81,920 | 16,384 | 12,288 |
| LDG / STG | 32,768 / 32,768 | 16,384 / 16,384 | 32,768 / 32,768 |
| **TOTAL** | **4,317,184** | **950,272** | **1,396,736** |

The grid is under one full wave (NCU launch rule: 0.86 waves for TileLang, 0.43
for cuTile), so there is no throughput amortization — dynamic instructions per
warp track static code size almost exactly (527 vs 680 static for TileLang, 116
vs 128 for Triton). **Static code bloat converts directly into duration here.**
That is why instruction counts and durations move together throughout.

---

## Cause 1: the index expression is re-materialized per syntactic use

`ir/tilelang_N1048576.cu` — the CUDA TileLang generates — is the direct
evidence. The loop body needs **two** division sites per element
(`offset // stride`, `offset % stride`) plus one `slice_1_offset // stage`.
TileLang emits, for a 2-element body:

- **74 distinct `rmod` locals and 40 distinct `rdiv` locals.** Every syntactic
  use of `slice_1_offset` re-expands the entire floordiv/floormod tree instead
  of reusing the computed value. There is no CSE at the TIR → C level.
- Each expansion carries the full **signed floor-division correction**, because
  `stride` is a runtime `int32` of unknown sign:
  `(((0 <= stride) && (0 <= rmod)) || ((stride < 0) && (rmod <= 0))) ? rdiv : rdiv - 1`
- A redundant third bound test per access, `... < 524288`, i.e. `padding_len/2`,
  on top of the user-written `< N` check.

Triton's frontend, given the identical expression, computes it once per element
and derives the rest.

## Cause 2: the tree is emitted a second time in int64, forcing a width dispatch

The same expression tree is re-emitted in **`int64_t`** for the buffer address —
12 further int64 division sites in the generated CUDA. NVVM lowers `div.s64`
into a runtime width dispatch: a 32-bit `div.u32` fast path plus a `div.s64`
slow path, guarded by `(offset|stride) & 0xFFFFFFFF00000000 != 0`. Visible in
`ir/tilelang_N1048576.ptx` at `$L__BB0_5`/`$L__BB0_6`, and in SASS as **4 real
`CALL` instructions** to a division subroutine.

PTX, high N:

| PTX opcode | TileLang | Triton |
|---|---:|---:|
| `div.s32` | 6 | 3 |
| `div.u32` | 4 | 0 |
| `div.s64` | 4 | 0 |
| `rem.s32` | 0 | 1 |
| **total div/rem sites** | **14** | 4 |
| `bra` | 28 | 0 |
| `ld.global` / `st.global` | 4 / 4 (32-bit) | 2 / 2 (**64-bit**) |
| **total instructions** | **317** | **46** |

## Cause 3: reciprocal CSE fails, so each site pays a full reciprocal chain

This is the core of the "extra division" observation.

**Triton computes one reciprocal per divisor and reuses it across numerators.**
In `analysis/triton_highN.sass_stream.txt` there are exactly two `MUFU.RCP` —
one for the `stride` divisor, one for `stage` — and each resulting reciprocal
feeds consecutive `IMAD.HI.U32 R*, <rcp>, <numerator>` for every element the
thread owns. It also strength-reduces `(x // stage) % 2 == 1` to a bit test
(`and.b32` with `0x80000001`; `setp.eq 1`), avoiding a division entirely.

**TileLang recomputes the whole chain at every site**: 11 `MUFU.RCP`, 11
`I2F.RP`, 11 `F2I`, each with its own `IABS` / `IMAD.HI` / correction sequence.
ptxas cannot merge them because (a) the expressions arrived as independent
locals (cause 1), (b) half are 64-bit and half 32-bit (cause 2), and (c) each
load and store sits inside its own guarded basic block (cause 4), so CSE has
nothing to hoist across.

Static SASS, high N:

| Opcode | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| Total instructions | **680** | 128 | 352 |
| IMAD (all forms) | **244** | 38 | 114 |
| ISETP | **113** | 18 | 70 |
| MUFU / I2F / F2I | **11 / 11 / 11** | 2 / 2 / 2 | 2 / 2 / 3 |
| IABS | 12 | 6 | 10 |
| BRA | **24** | 1 | 1 |
| BSSY / BSYNC | **9 / 9** | 0 / 0 | 0 / 0 |
| CALL (64-bit div subroutine) | **4** | 0 | 0 |
| LDG / STG | 4 / 4 (`.E`) | 2 / 2 (`.E.64`) | 8 / 8 (`.E`) |

The NCU side: `MUFU` 81,920 vs 16,384 dynamic warp instructions — a **5x**
ratio, against 4.5x for total instructions. And XU-pipe activity, where
`MUFU`/`I2F`/`F2I` execute, is 29.4% for TileLang and falls to **0.0%** in the
probe below when division is removed. The XU column is a pure division meter.

## Cause 4: per-access guard branches instead of predication

TileLang lowers `if valid_1:` and `T.if_then_else` into taken branches, up to
three nested per access (the `< N` test, a negative-index test, and the
`< 524288` buffer bound). Triton lowers `mask=` into predicated loads and
stores. Dynamically: **188,416 `BRA` plus 147,456 `BSSY`/`BSYNC`**
reconvergence instructions for TileLang, against **zero** of either for both
other backends — about 7.8% of TileLang's instruction count spent on control
flow that does not exist elsewhere.

But the probe below shows removing the guards alone costs **5.5% of instructions
and 0% of duration** — 9.472 us either way. The branches are issue-cheap and
fully predictable. Their real cost is indirect: the basic-block fragmentation is
what blocks the reciprocal CSE in cause 3. Only once division is gone does guard
removal show up in time at all (6.336 → 6.080 us, 4.0%).

## Cause 5: weaker vectorization — 2x the memory instructions for identical traffic

All three backends move **identical sector traffic**: 131,072 global-load
sectors, 131,072 store sectors, ~4.20 MB DRAM read. But:

| | L1 ld requests | sectors/request | SASS |
|---|---:|---:|---|
| TileLang | 32,768 | 4 | 4x `LDG.E` (32-bit) |
| Triton | **16,384** | **8** | 2x `LDG.E.64` |
| cuTile | 32,768 | 4 | 8x `LDG.E` (32-bit) |

TileLang and cuTile need twice as many memory instructions to move the same
bytes. This is the residual after causes 1–4, and it is the whole remaining gap
between the best probe variant (6.080 us) and Triton (5.856 us). See the
specialization caveat below — this one is not purely a codegen-quality
difference.

---

## Attribution probe

To separate the causes I compiled four TileLang variants of the *same* kernel,
changing one thing at a time, and profiled each identically. **These are
diagnostic instruments, not proposed implementation changes** — see the parity
note below. All four verify bit-exact against a PyTorch reference
compare-exchange step (`harness/profile_variant.py --verify`).

| Variant | What changed | Duration | Warp instructions |
|---|---|---:|---:|
| `v0_baseline` | nothing (as shipped) | 9.472 us | 4,317,184 |
| `v3_novalid_only` | guards removed, div/mod kept | 9.472 us | 4,079,616 |
| `v1_shift` | div/mod → shift/mask, guards kept | **6.336 us** | **1,269,760** |
| `v2_shift_novalid` | both | 6.080 us | 999,424 |
| Triton | reference | 5.856 us | 950,272 |
| cuTile | reference | 6.272 us | 1,396,736 |

Isolating the division mechanism (causes 1–3) removes **91% of the instruction
excess over Triton and 87% of the duration gap**. Isolating the guards
(cause 4) removes 5.5% of instructions and no measurable time. Supporting
counters:

| | v0_baseline | v1_shift | v2_shift_novalid | Triton |
|---|---:|---:|---:|---:|
| XU active | 29.0% | **0.0%** | **0.0%** | 19.1% |
| MUFU / I2F / F2I | 81,920 each | **0** | **0** | 16,384 each |
| BRA | 188,416 | 65,536 | **0** | 0 |
| BSSY + BSYNC | 147,456 | 65,536 | **0** | 0 |
| IMAD | 1,359,872 | 196,608 | 131,072 | 311,296 |
| ISETP | 851,968 | 131,072 | 32,768 | 147,456 |
| L1 ld requests | 32,768 | 32,768 | 32,768 | **16,384** |
| Static SASS instructions | 680 | 168 | **136** | 128 |

Read the bottom two rows together. Once division is gone, TileLang's static code
is within 6% of Triton's (136 vs 128 instructions) and it executes 5% more
instructions — yet it is still 3.8% slower, because the L1 request count never
moves. That is cause 5 in isolation, and it confirms the causes are
independent: arithmetic bloat and memory-instruction count contribute
separately.

### Parity note

`v1`/`v2` pass `log_stride`/`log_stage` and index with shifts and masks. Triton
and cuTile keep `//` and `%` in their sources, so a shift/mask TileLang kernel
would **not** be the same implementation — it would be a hand-specialization of
one backend against two unspecialized ones. That is why these variants stay in
`harness/` and are used only to attribute cost. The parity-preserving reading of
the probe is: *given the same source arithmetic, 91% of TileLang's excess is
recoverable by the compiler alone* — a reciprocal CSE plus keeping int32 index
math in int32 would get there without touching the operator.

The one place source parity is genuinely imperfect: TileLang's `if valid_1:`
against Triton's `mask=` argument is a small idiom difference rather than
identical source. The probe bounds its effect at zero measurable time, so it
does not affect the conclusion.

---

## Two caveats on the comparison

**Triton's vectorization is a specialization, not better analysis.** The wide
access path comes from Triton's `divisible_by_16` argument specialization.
Recompiling the same kernel at each stride (`ir/triton_N1048576_stride*.ptx`,
captured at `num_warps=4`, where the effect is larger — 128-bit rather than the
profiled config's 64-bit accesses):

| stride | PTX instructions | Global accesses | div/rem sites |
|---:|---:|---|---:|
| 16 | **78** | 2x `ld.global.v4.b32`, 2x `st.global.v4.b32` | 8 |
| 8 | 111 | 8x `ld.global.b32`, 8x `st.global.b32` | 8 |
| 4 | 111 | 8 + 8 scalar | 8 |
| 2 | 111 | 8 + 8 scalar | 8 |
| 1 | 93 | 8 + 8 scalar | 4 |

Only `stride % 16 == 0` gets a vectorized path. Across a full N = 2^20 sort, 74
of the 210 launches run at stride < 16 and take the scalar path, so ~35% of
Triton's launches are slower than this pass suggests. TileLang and cuTile take
runtime `stride` with no divisibility specialization and are always scalar.
Cause 5 is therefore partly a JIT-specialization-policy difference, not only
codegen quality.

**cuTile's gap is a source-level API choice, not codegen.** cuTile avoids the
repeated division bundles (2 `MUFU.RCP`, same as Triton) and its arithmetic is
clean. Its 1.397 M instructions come from the `ct.gather` / `ct.scatter` path —
8 static `LDG.E` and 8 `STG.E` against Triton's 2 + 2 — plus the defensive
two-step OOB handling in `impl_cutile.py` (16 `FSEL`, 8 `VIMNMX`, 3 `BAR`).
Unlike causes 1–5, that is written in the operator, so cuTile's position in
this comparison is not an apples-to-apples compiler measurement.

---

## Low N

| Metric | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| Duration | 5.888 us | **4.320 us** | 5.504 us |
| Executed warp instructions | **16,864** | 3,712 | 5,160 |
| Grid / block | 4 / 256 | 4 / 256 | 2 / 128 |
| MUFU | 320 | 64 | 16 |
| BRA | 736 | 0 | 0 |

Four CTAs cannot fill 148 SMs; the ~5 us floor is launch and underfill. The
instruction ratios reproduce (4.5x vs Triton, 5x on `MUFU`), but the utilization
percentages carry no information at this duration. cuTile runs a 2-CTA grid here
because its low-N winner is `tile=1024`.

## NCU's own rules

The rule engine adds nothing beyond launch underfill — it flags 0.86 full waves
for TileLang and 0.43 for cuTile, an L2-compression note worth an estimated
4.2%, and (for cuTile) a spurious "tensor core pipe utilization is 0%" on an
integer-indexing kernel. Full output in `analysis/details_*_highN.txt`.

## Reproducing

```bash
source harness/env.sh            # env python, PYTHONPATH, nvcc -ccbin, GPU pin
$PY harness/dump_ir.py --outdir ir              # PTX / SASS / cubin / TileIR
bash harness/run_ncu.sh                         # 12 backend reports
$PY harness/profile_variant.py --variant v1_shift --verify   # probe check
$PY harness/analyze_ir.py                       # analysis/static_ir_mix.txt
$PY harness/analyze_ncu.py                      # analysis/ncu_metrics.{txt,json}
$PY harness/analyze_variants.py                 # analysis/variant_comparison.txt
```

cuTile needs `CUDA_TILE_TEMP_DIR`, `CUDA_TILE_DUMP_TILEIR=1`,
`CUDA_TILE_DUMP_BYTECODE=1` and a *fresh* `CUDA_TILE_CACHE_DIR` for
`dump_ir.py` to capture its JIT cubin — on a cache hit it never writes one.
There is no PTX stage to dump, since TileIR lowers straight to cubin.
