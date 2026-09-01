# sigmoid — why TileLang loses at fp16/bf16 but wins at fp32 on B200

N=50,000,000, B200 sm_100, 148 SMs, NCU 2026.1.1, idle GPU pinned by UUID.
All 9 captures verified against `torch.sigmoid`. Method:
`kernel-perf-differential` skill.

## Verdict

**TileLang executes ~1.44x Triton's instructions for the same elementwise work,
at both dtypes. At fp32 that is free because DRAM saturates and hides it; at
fp16 the memory demand halves, the overhead becomes binding, and it costs 1.43x
the cycles.**

The gap is *not* a bandwidth problem, a vectorization problem, or a
transcendental-math problem — all three are identical across backends. It is a
pure instruction-count problem that is only *visible* at narrow dtypes.

The decomposition closes to 0.1%:

```
1.4374x instructions / 1.0062x issue rate = 1.4286x cycles   (measured 1.4303x)
```

## The dtype signature

Benchmark medians over 20 shapes (`results/b200_autotuned_postmerge`):

| dtype | TL/Triton | TL/cuTile |
|---|---:|---:|
| fp32 | **1.02** | 1.02 |
| fp16 | 1.39 | 0.98 |
| bf16 | 1.39 | 0.98 |

At N=50M, achieved bandwidth tells the story before any profiling:

| dtype | TL TB/s | Triton TB/s | TL Gelem/s | Triton Gelem/s |
|---|---:|---:|---:|---:|
| fp32 | **6.66** | 6.57 | 833 | 821 |
| fp16 | 4.23 | **6.17** | **1059** | **1543** |

TileLang reaches 6.66 TB/s at fp32 — near HBM peak, and *faster than Triton*.
So it is not fundamentally slow. But when bytes halve, Triton scales 821 -> 1543
Gelem/s (**1.88x**, near the ideal 2x) while TileLang manages 833 -> 1059
(**1.27x**). TileLang hits a per-element ceiling around ~1050 Gelem/s that
Triton does not.

## What is identical — the eliminations

Measured at matched `BLOCK_SIZE=2048`, so the search-space difference (TileLang's
tuner caps at 2048; Triton and cuTile go to 8192) cannot contaminate anything.

| | TileLang | Triton |
|---|---:|---:|
| `MUFU` (the sigmoid itself) | 3,125,008 | 3,125,120 |
| `xu` pipe, absolute | 42,230 | 42,231 |
| `LDG` | 195,313 | 195,320 |
| `STG` | 195,313 | 195,313 |
| load/store width | `LDG.E.128` / `STG.E.128` | `LDG.E.128` / `STG.E.128` |
| issued warps / scheduler | 0.7466 | 0.7420 |

Essential work, memory instruction count, access width and issue rate are all
equal. **The entire gap is surrounding instructions.**

## Where the extra instructions are

Total dynamic instructions, matched config:

| | TileLang | Triton | ratio |
|---|---:|---:|---:|
| fp16 | 35,937,648 | 25,000,946 | **1.44x** |
| fp32 | 35,547,022 | 23,829,024 | **1.49x** |

Note the totals are nearly *identical across dtypes* for each backend. The
overhead is not dtype-specific in magnitude — but its composition is completely
different, which is the interesting part.

### fp16: conversion and half-precision packing

| opcode | TileLang | Triton | extra |
|---|---:|---:|---:|
| `F2FP` (fp32<->fp16 convert) | **3,906,260** | 781,280 | +3,124,980 |
| `HADD2` | 3,906,260 | 1,562,560 | +2,343,700 |
| `SHF` | 1,953,137 | 195,320 | +1,757,817 |
| `PRMT` (byte permute) | **976,565** | **0** | +976,565 |
| `HFMA2` | 976,565 | 195,320 | +781,245 |

`F2FP` alone is **5.0x** Triton's count. Conversion and half-precision
packing (`F2FP`+`HADD2`+`PRMT`+`HFMA2`) account for roughly **73%** of the
~10.9M instruction gap.

Root cause in source: Triton and cuTile both hoist the conversion to the tile
boundary — `tl.load(...).to(tl.float32)` and `ct.astype(x, ct.float32)` — so
they convert once per tile and compute in fp32. TileLang applies
`T.sigmoid(x[idx])` **per element on the native dtype**, so conversion and
repacking happen per element.

This also shows up numerically: TileLang's `max_abs_err` is **6.999e-04** versus
**2.441e-04** for both competitors — 2.9x larger, consistent with computing in
fp16 rather than fp32.

### fp32: divergent branch machinery

| opcode | TileLang | Triton |
|---|---:|---:|
| `BSSY` | **1,562,504** | **0** |
| `BSYNC` | **1,562,504** | **0** |
| `BRA` | **1,562,504** | **0** |
| `LOP3` | 1,953,137 | 195,320 |
| `SHF` | 1,953,137 | 195,320 |
| `ISETP` | 1,757,824 | 390,640 |
| `IADD3` | 1,562,504 | 195,320 |

At fp32 there are no conversions to pay for, so the overhead comes from
somewhere else entirely: **4.7M branch-stack instructions that Triton does not
emit at all**, plus ~5.5M extra integer/index operations.

That is the `if idx < n_elements` guard in TileLang's kernel compiled as a real
divergent branch, plus per-element index arithmetic:

```python
for local_idx in T.Parallel(BLOCK_SIZE):
    idx = local_idx + pid * BLOCK_SIZE
    if idx < n_elements:
        output[idx] = T.sigmoid(x[idx])
```

Triton's `mask=mask` becomes predication on the load and store — no branch, no
per-element index recomputation.

**The common root across both dtypes is the per-element scalar formulation.**
Its cost just surfaces through different opcodes depending on dtype.

## Why it only matters at fp16

Nothing is saturated at fp16 — this is a latency-bound regime and all three
backends are ~93-94% stalled with similar distributions, so stall composition
is uninformative (the helper correctly routes to step 5b here).

The binding constraint is DRAM headroom:

| | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| DRAM % peak, fp32 | **78.94** | **80.73** | 73.67 |
| DRAM % peak, fp16 | 43.54 | 58.38 | 34.93 |
| issue/cycle, fp32 | 0.61 | **0.41** | 0.70 |
| issue/cycle, fp16 | 0.75 | 0.74 | 0.71 |

At fp32 both are near DRAM peak. Triton's issue rate drops to **0.41** — it is
idling on memory, and TileLang simply fills those idle slots with its extra
instructions. Cycles tie (99,585 vs 98,388) despite TileLang running 1.49x the
instructions.

At fp16 the byte demand halves. Triton's issue rate rises to 0.74 and it
finishes early. TileLang still has to execute the same ~35.9M instructions at
the same 0.75 issue rate, so it takes 1.43x the cycles. The overhead was always
there; fp32 just paid for it out of memory-stall slack.

## Ordering check

Instruction count predicts the ranking exactly, at both dtypes:

| | instructions | duration (tuned fp16) |
|---|---:|---:|
| Triton | 23.6M | 30,752 ns |
| TileLang | 35.9M | 46,944 ns |
| cuTile | 41.9M | 51,200 ns |

cuTile has *more* instructions than TileLang and is correspondingly slower —
which is why cuTile also loses to Triton here, and why TileLang and cuTile are
near parity (0.98x) in the benchmark.

## What is established, and what is not

**Established:** identical essential work, memory instructions, access widths
and issue rates; a 1.44-1.49x instruction gap at both dtypes; the exact opcode
composition of that gap at each dtype; the cycles identity closing to 0.1%; and
that DRAM saturation at fp32 is what hides the overhead.

**Not established:** that fixing the formulation would recover the full gap. No
intervention was run. The predicted fix is to rewrite the TileLang kernel in
tile form — load a tile, convert once, compute in fp32, convert once, store with
a mask instead of an `if` — mirroring what Triton and cuTile already do. That is
a small, structurally faithful change (unlike gaussian_blur, where matching the
competitors would require a full rewrite), so it is a legitimate experiment
rather than a fairness violation.

**Secondary, untested:** TileLang's autotuner caps `BLOCK_SIZE` at 2048 while
Triton and cuTile search to 8192, and 29 of 60 tuned winners sit exactly at that
cap. Matching the config changed the gap only from 1.53x to 1.36x, so the cap is
a minor contributor at most — but it should be widened before any re-measurement.

## Relation to the other operators

This is the second operator where TileLang is fine at fp32 and degrades at a
narrower dtype — `matrix_transpose` is at parity in fp32 and loses 1.86x at
INT8 (bank conflicts from a missing dtype-aware swizzle). Different mechanism,
same shape: **TileLang's generic lowering happens to be adequate at 4-byte
elements and leaves work on the table below that.** Two independent operators is
suggestive, not conclusive.

## Reproducing

```bash
cd profile/b200_autotuned_postmerge/sigmoid
source harness/env.sh
bash harness/run_ncu.sh          # 9 captures, ~2 min

PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python:$PYTHONPATH \
  $PY ~/.claude/skills/kernel-perf-differential/helpers/sass_opcodes.py \
    reports/src_tilelang_fp16.ncu-rep:tilelang \
    reports/src_triton_fp16.ncu-rep:triton
```

## Artifacts

- `harness/profile_sigmoid.py` — single-launch, config-forced, correctness-checked.
- `harness/run_ncu.sh` — 9 captures: tuned fp16, tuned fp32, matched fp16.
- `reports/tuned_*`, `reports/match_*` — counter captures (8 sections).
- `reports/src_*` — source-counter captures for dynamic opcode histograms.
- `analysis/sass_opcodes_fp16.txt`, `analysis/sass_opcodes_fp32.txt`
- `analysis/stall_match_fp16.txt`, `analysis/stall_tuned_fp32.txt`
