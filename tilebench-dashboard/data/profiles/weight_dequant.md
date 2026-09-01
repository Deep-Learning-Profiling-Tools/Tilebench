# weight_dequant — why cuTile is slower than Triton and TileLang on B200

fp16, M=N=8192, TILE_SIZE=128. B200 sm_100, 148 SMs, NCU 2026.1.1, idle GPU
pinned by UUID. All three backends verified `max_abs_err = 0.000000`.
Method: `kernel-perf-differential` skill.

## Verdict

**`ct.gather` turns a bandwidth-bound kernel into an ALU-bound one.**

cuTile issues **7.08x TileLang's instructions**, saturates the integer ALU pipe
at **85.6% of peak**, and as a consequence reaches only **31.6% of DRAM peak**
against Triton's 64.1% and TileLang's 67.9% — on a kernel whose entire job is
moving bytes. It is not slow at moving data; it never gets to move data.

Unlike the matmul and BMM investigations, here a pipe is **genuinely
saturated**, so this is a throughput bottleneck rather than an
instruction-count-in-a-latency-bound-regime story.

In the recorded postmerge results cuTile is median **1.82x** the best backend
(worst 2.91x); Triton 1.00x and TileLang 1.01x are tied. torch is 12.70x and
irrelevant here.

## Step 1 — the implementations are structurally identical

All three flatten to a 1-D tile and recover 2-D coordinates the same way:

```python
# Triton                                  # cuTile                       # TileLang
row = offsets // N                        row = offsets // N             row = idx // N
col = offsets % N                         col = offsets % N              col = idx % N
s_row = row // TILE_SIZE                  s_row = row // TILE_SIZE       s_row = row // TILE_SIZE
s_col = col // TILE_SIZE                  s_col = col // TILE_SIZE       s_col = col // TILE_SIZE
```

**The index arithmetic is not the differentiator — everyone does it.** The only
line that differs is how the scale is fetched:

| backend | scale fetch |
|---|---|
| Triton | `tl.load(S + s_row * S_COLS + s_col, mask=mask)` |
| TileLang | direct indexed read `S[s_row, s_col]` |
| **cuTile** | **`ct.gather(s_ptr, (s_row, s_col), padding_value=0)`** |

Launch geometry is identical for all three: **grid 65,536, block 128**.

## Measurements

| | Triton | TileLang | cuTile |
|---|---:|---:|---:|
| duration (ns) | 45,696 | **42,784** | **93,344** (2.18x) |
| `sm__cycles_active.avg` | 79,131 | 73,748 | 168,705 |
| **instructions / SM** | 113,360 | **69,078** | **488,863 (7.08x)** |
| inst / cycle | 1.433 | 0.937 | 2.898 |
| **registers / thread** | 22 | 20 | **37** |
| **static shared memory** | **0 B** | **0 B** | **4,108 B** |
| **DRAM % of peak** | 64.1 | **67.9** | **31.6** |
| achieved bandwidth | 4.92 TB/s | 5.20 TB/s | 2.42 TB/s |

## The mechanism: ALU saturation

Pipe utilization, % of peak sustained active (a **rate**):

| pipe | Triton | TileLang | cuTile | CT/TL |
|---|---:|---:|---:|---:|
| **`alu`** | 34.695% | 14.411% | **85.567%** | **5.9x** |
| `lsu` | 24.622% | 9.607% | 30.447% | 3.2x |
| **`uniform`** | 0.000% | 0.000% | **9.974%** | inf |
| `xu` | 0.000% | 0.000% | 2.100% | inf |
| `fma` | 7.834% | 9.607% | 11.549% | 1.2x |
| `adu` | 17.999% | 24.110% | 10.526% | 0.4x |
| `cbu` | 1.119% | 0.600% | 1.050% | 1.7x |

Same rows as absolute instruction volume per SM (rate x cycles):

| pipe | Triton | TileLang | cuTile | CT/TL |
|---|---:|---:|---:|---:|
| **`alu`** | 27,454 | 10,627 | **144,356** | **13.6x** |
| `lsu` | 19,484 | 7,085 | 51,366 | 7.2x |
| `uniform` | 0 | 0 | 16,827 | inf |
| `fma` | 6,199 | 7,085 | 19,484 | 2.8x |
| `xu` | 0 | 0 | 3,542 | inf |
| `adu` | 14,243 | 17,780 | 17,757 | 1.0x |

**`alu` at 85.567% of peak is the finding.** Note `adu` (address generation) is
*identical in absolute terms* across TileLang and cuTile — the extra work is not
address generation, it is integer ALU. cuTile is also the only backend touching
the `uniform` and `xu` pipes at all.

## Stall composition confirms it — and it differs qualitatively

Normalized by resident warps:

| stall | cuTile | Triton | TileLang |
|---|---:|---:|---:|
| `long_scoreboard` (global/L1 data wait) | **19.0%** | 69.7% | **86.2%** |
| `math_pipe_throttle` (math pipe busy) | **23.5%** | 2.0% | 0.4% |
| `not_selected` (another warp won) | **27.6%** | 2.9% | 0.7% |
| `wait` (arithmetic dependency) | 7.9% | 5.3% | 3.3% |
| `barrier` | 6.5% | 0.0% | 0.0% |
| **TOTAL stalled** | 92.8% | 94.8% | 97.3% |

Triton and TileLang are **memory-bound**: 70-86% of stall time waiting on data,
which is the correct shape for a dequantization kernel. cuTile spends only 19%
waiting on memory and instead splits between **math-pipe saturation (23.5%, 58x
TileLang)** and **issue contention (27.6%)**.

`math_pipe_throttle` is the counter that says the math pipe is *genuinely full*,
and it corroborates the 85.6%-of-peak ALU reading from an independent direction.

## Scheduler: cuTile is oversubscribed, not starved

| per scheduler | cuTile | Triton | TileLang |
|---|---:|---:|---:|
| active warps | 10.648 | 11.415 | 10.731 |
| **eligible warps** | **3.668** | 0.698 | 0.304 |
| issued warps | **0.725** | 0.359 | 0.235 |
| % cycles with no eligible warp | **27.5%** | 64.1% | 76.5% |

cuTile has **12x TileLang's eligible warps** and issues **3.08x** as often. It is
working harder than either competitor and losing anyway. The identity closes
exactly:

```
7.08x instructions / 3.08x issue rate = 2.29x cycles     (measured 2.29x)
```

## Why a gather is the wrong primitive here

The access pattern is maximally structured: **every element in a 128x128 block
wants the same scale value**. Triton and TileLang compute one flat address and
issue an ordinary load, which coalesces to a single sector per warp. `ct.gather`
is compiled for arbitrary per-lane addresses and cannot assume that, so it pays:

- explicit tile/index handling -> the 13.6x ALU volume
- `padding_value=0` bounds logic -> part of the same
- staging -> **4,108 B of static shared memory**, versus zero for both others
- higher register pressure -> **37 vs 20-22** registers/thread

Cache behaviour is unremarkable and does not explain anything: L1 sector hit
19.68% (cuTile) vs 19.71% (Triton) vs 2.66% (TileLang); L2 hit ~0.25% for all.

## What is established, and what is not

**Established:** cuTile executes 7.08x the instructions, concentrated 13.6x in
the integer ALU; the ALU pipe is at 85.6% of peak; `math_pipe_throttle` is 23.5%
of stall time; DRAM utilization is halved as a result. All three kernels are
structurally identical apart from the scale-fetch primitive, at identical launch
geometry, with identical numerical output.

**Not established:** that `ct.gather` *specifically* is responsible, as opposed
to something else in cuTile's lowering. The evidence is circumstantial-by-
elimination — it is the one line that differs, and cuTile is the outlier on
instructions, registers and shared memory. **No intervention was run.**

The decisive test is one line: replace `ct.gather(s_ptr, (s_row, s_col))` with a
flat computed-address `ct.load`, and re-measure. If ALU utilization drops and
DRAM utilization rises toward 65%, the case is closed.

## Perspective

This operator is small (2-7 us at benchmark shapes) and all three DSLs beat
torch by 12.7x. The 1.82x median cuTile deficit is real but low-stakes compared
with the GEMM operators.

## Reproducing

```bash
source harness/env.sh
bash harness/run_ncu.sh          # 3 backends, 7 sections, ~1 min
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python:$PYTHONPATH \
  $PY ~/.claude/skills/kernel-perf-differential/helpers/stall_breakdown.py \
    reports/cutile_fp16.ncu-rep:cutile \
    reports/triton_fp16.ncu-rep:triton \
    reports/tilelang_fp16.ncu-rep:tilelang
```

## Note on tooling

`stall_breakdown.py` printed its "every backend is >85% stalled, composition
cannot explain the gap" warning on this data, and **that heuristic is wrong
here**. Composition differs *qualitatively* — memory-wait for two backends,
math-throttle and issue contention for the third — and that difference is
exactly what identifies the cause. The heuristic was calibrated on the matmul
case where all backends shared the same stall shape. It should additionally
check whether the stall *distributions* diverge before advising the reader to
move on.
