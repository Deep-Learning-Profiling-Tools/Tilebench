# Matrix Transpose: B200 NCU Profile

## Verdict

TileLang's INT8 transpose gap is caused by the shared-memory address layout
generated for `T.transpose`. It is not caused by global-memory coalescing,
insufficient occupancy, or total instruction count.

At `4096 x 20480`, using each backend's recorded autotune winner:

| dtype | TileLang | Triton | cuTile | TL/Triton | TL/cuTile |
|---|---:|---:|---:|---:|---:|
| FP16 benchmark | 59.399 us | 53.900 us | 54.208 us | 1.10x | 1.10x |
| INT8 benchmark | 60.916 us | 30.534 us | 36.295 us | **2.00x** | **1.68x** |
| FP16 NCU | 57.600 us | 50.048 us | 50.272 us | 1.15x | 1.15x |
| INT8 NCU | 63.296 us | 30.912 us | 39.488 us | **2.05x** | **1.60x** |

Triton becomes 38.2% faster when moving from FP16 to INT8, and cuTile becomes
21.5% faster. TileLang instead becomes 9.9% slower despite moving half as many
logical bytes. NCU localizes the lost scaling to `T.transpose`'s shared stores:

- TileLang INT8: **8.588M shared-store bank conflicts**.
- Triton INT8: 0.255M.
- cuTile INT8: 0.011M.
- TileLang's short-scoreboard stall rises to **50.52 cycles per issued
  instruction**, versus 2.04 for Triton and 0.88 for cuTile.

## Setup

- GPU: NVIDIA B200, SM 10.0.
- Nsight Compute: 2026.1.1.
- Shape: `M=4096`, `N=20480`.
- Dtypes: FP16 and INT8.
- Collection: one direct transpose launch after three warmups; `--set full`
  plus PM sampling, and separate `--set source --section SourceCounters` runs.
- Validation: all six isolated launches matched `torch.equal(output, x.T)` and
  produced matching per-dtype checksums.
- Benchmark source:
  `results/runs/20260720_045231_b200tuned_resume/operators/matrix_transpose.json`.

Recorded autotune winners:

| dtype | TileLang | Triton | cuTile |
|---|---|---|---|
| FP16 | `tile=128, threads=128` | `tile=64, warps=8` | `tile=128, occupancy=4` |
| INT8 | `tile=64, threads=128` | `tile=128, warps=8` | `tile=128, occupancy=4` |

The different tiles are intentional: this profile compares the actual winners,
not forced common configurations.

## INT8 Headline Counters

| Metric | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| Duration | **63.296 us** | 30.912 us | 39.488 us |
| Logical read+write bandwidth | **2.65 TB/s** | 5.43 TB/s | 4.25 TB/s |
| Grid blocks | 20,480 | 5,120 | 5,120 |
| Block threads | 128 | 256 | 128 |
| Registers/thread | 21 | 32 | 113 |
| Achieved occupancy | **91.38%** | 96.74% | 25.54% |
| SM throughput | 32.21% | 52.17% | 46.75% |
| Memory throughput | **89.70%** | 69.85% | 58.46% |
| Shared instructions | **5.407M** | 3.277M | 2.785M |
| Shared wavefronts | **14.887M** | 3.644M | 3.337M |
| Shared bank conflicts | **8.668M** | 0.282M | 0.014M |
| Short-scoreboard stall | **50.52** | 2.04 | 0.88 |
| MIO-throttle stall | **32.22** | 8.72 | 0.04 |
| Executed warp instructions | **9.093M** | 13.681M | 15.278M |

TileLang executes fewer instructions and has high occupancy. Nevertheless, its
schedulers have no eligible warp in 86.6% of cycles because the active warps
are waiting on shared-memory dependencies. Adding more warps cannot solve this
specific bottleneck.

NCU reports 89.7% memory throughput for TileLang, but only 26.6% DRAM
throughput. The saturated unit is L1/shared memory, not HBM.

## Bank-Conflict Evidence

All three backends emit scalar `STS.U8` instructions during the INT8
rearrangement. Scalar byte stores are therefore not sufficient to explain the
gap. Their address mappings differ:

- TileLang's representative `STS.U8` accesses require four shared-memory
  wavefronts and report three excessive wavefronts per request.
- Triton's and cuTile's representative `STS.U8` accesses require one
  wavefront, with no conflict on those instructions.

Dynamic totals make the effect visible:

| Counter | TileLang INT8 | Triton INT8 | cuTile INT8 |
|---|---:|---:|---:|
| Shared-store instructions | 2.785M | 2.621M | 2.621M |
| Shared-store wavefronts | **11.865M** | 2.876M | 2.632M |
| Shared-store bank conflicts | **8.588M** | 0.255M | 0.011M |

TileLang performs only 6.3% more shared-store instructions than Triton, but
generates 4.13x as many store wavefronts and 33.7x as many bank conflicts.
Against cuTile, the conflict count is almost 795x higher.

NCU's rule engine estimates that TileLang's shared-store conflicts represent
69.55% local improvement potential. Its broader source-counter rule finds
7.864M excessive shared wavefronts, or 57% of the total, with 53.34% estimated
improvement potential. These estimates overlap and must not be added.

### Why INT8 is affected much more than FP16/FP32

NVIDIA shared-memory banks are addressed at 4-byte granularity. In simplified
form, the bank selected by a byte address is:

```text
bank = (byte_address / 4) % 32
```

The generic TileLang transpose assigns adjacent logical destination elements
to adjacent lanes without a dtype-aware bank swizzle. Therefore, for a
contiguous portion of the transposed destination:

- FP32: one 4-byte value fills one bank word, so adjacent lanes naturally move
  to adjacent banks.
- FP16/BF16: two adjacent 2-byte values share a bank word, so the mapping can
  create a two-lane collision.
- INT8: four adjacent 1-byte values share a bank word, so four lanes can target
  different bytes in the same bank during the same `STS.U8` instruction.

The INT8 SASS confirms this exact case: representative TileLang `STS.U8`
instructions report four wavefronts with three excessive wavefronts. The
corresponding Triton and cuTile `STS.U8` instructions report one wavefront.
Their generated lane/address mappings are therefore effectively permuted or
swizzled so adjacent logical byte elements do not imply four simultaneous
accesses to the same bank. The profiles prove the resulting mapping; they do
not prove that either compiler uses an internal pass literally named
"swizzle."

FP16 is not completely conflict-free: TileLang records 0.703M shared-store
conflicts and 8% excessive shared wavefronts. INT8 increases this to 8.588M
conflicts and 57% excessive wavefronts. Two additional effects amplify the
element-width problem:

1. The tuned TileLang INT8 kernel uses a 64x64 tile instead of FP16's 128x128
   tile, so it launches four times as many CTAs and output TMA operations.
2. TileLang INT8 executes 3.49x as many dynamic shared-store instructions as
   TileLang FP16. The narrower type does not become a proportionally wider or
   better-packed shared-memory operation.

Finally, INT8 halves the useful global-memory bytes. Triton and cuTile benefit
from that reduction, while TileLang merely exposes the now-dominant shared
transpose cost. This is why FP16 can remain DRAM-facing and roughly competitive
while INT8 becomes L1/shared-memory-bound and fails to speed up at all.

## Source Attribution

The hottest TileLang INT8 source-counter location is generated
`tvm_kernels.cu:34`, with:

- 1,866 short-scoreboard samples.
- 221 MIO-throttle samples.
- 208 barrier samples.

The SASS at that location is the repeated `LDS.U8` / conflict-heavy `STS.U8`
sequence implementing the call to `T.transpose` at
`benchmarks/operators/matrix_transpose/impl_tilelang.py:36`.

The input-load location is secondary, with 584 long-scoreboard samples. The
final shared-to-global copy lowers to `UTMASTG.2D`; it has only 21
long-scoreboard samples and is not the main problem. TileLang's global loads
also produce exactly 2.621M L1 sectors, the same as Triton and cuTile, so the
input path is correctly coalesced.

## Why TileLang Generates This Layout

The installed TileLang implementation confirms the generic lowering path:

- `src/op/transpose.cc:174-204` creates a SIMT nested parallel loop that loads
  `src[i,j]` and stores `dst[j,i]`.
- `TransposeNode::InferLayout` at lines 211-215 returns an empty layout map and
  states that no special layout inference is needed.
- `src/cuda/op/transpose.cc:19-24` registers the common backend transpose
  lowering for CUDA.
- `src/backend/common/op/transpose.h:38-56` delegates the fused parallel loop
  to generic `ParallelOp` layout inference.

That generic mapping is adequate for larger element types, but it is not
bank-aware for byte stores. Four neighboring INT8 accesses repeatedly land on
the same 4-byte shared-memory bank, creating the observed conflict pattern.

## FP16 Comparison

TileLang FP16 is a different, smaller issue:

- Duration is 57.6 us versus roughly 50.1 us for both competitors.
- Its 128x128 kernel allocates 64 KiB dynamic shared memory and reaches only
  15.3% occupancy.
- It remains primarily DRAM-facing at 64.7% memory/DRAM throughput.
- It has 0.865M shared conflicts, but only 8% excessive shared wavefronts,
  versus 57% for INT8.

This explains why FP16 remains within about 10-15% while INT8 collapses. The
FP16 kernel pays shared-memory capacity and ordinary memory-latency costs; the
INT8 kernel saturates the shared-memory path with conflict replay.

## Recommended Fix Order

1. **Add a bank-aware CUDA lowering for `T.transpose`, especially for 8-bit
   elements.** Use a swizzled/padded destination layout or a warp mapping that
   distributes adjacent byte stores across banks. This is the real compiler
   fix because `InferLayout` currently provides no transpose-specific layout.
2. **Consider packing four INT8 elements before the shared store.** A
   `uint32`/vectorized permutation can replace conflicting independent byte
   stores, provided the mapping preserves coalesced global access. Triton and
   cuTile show that scalar stores can also work when laid out correctly, so
   packing is an option rather than a requirement.
3. **Retune tile size after fixing the layout.** TileLang currently selects a
   64x64 tile and launches 4x as many CTAs and TMA stores as the 128x128
   competitors. The autotuner is likely avoiding an even worse 128x128
   conflict/capacity tradeoff. A conflict-free transpose may make 128x128
   competitive.
4. **Treat the FP16 optimization separately.** Reducing the two-shared-tile
   footprint or using a one-tile transpose could improve its 15.3% occupancy,
   but it will not fix the INT8 bank mapping by itself.

A successful INT8 fix should reduce shared-store conflicts from 8.588M toward
Triton's 0.255M or below, bring short-scoreboard stall well under 5 cycles, and
move runtime toward the 31-40 us competitor range.

## Caveats

- The reports compare autotune winners, so tile and launch geometries differ.
  The diagnosis uses per-access counters and FP16-to-INT8 scaling in addition
  to absolute totals to account for this.
- NCU replay perturbs timing, but the NCU ratios closely reproduce the original
  benchmark ratios and the hardware-counter diagnosis does not depend on
  timing alone.
- TileLang's temporary `tvm_kernels.cu` could not be imported after JIT
  compilation. NCU retained source line mappings, SASS, and all source-counter
  data.
- No matrix-transpose implementation code was changed during this run.

## Reproduction

```bash
cd /home/arustagi/repos/Tilebench
RUN=profile/b200_autotuned_postmerge/matrix_transpose
NCU=/opt/nvidia/nsight-compute/2026.1.1/ncu
PY=/home/arustagi/anaconda3/envs/tilebench/bin/python

$NCU --set full --section PmSampling --section PmSampling_WarpStates \
  --import-source on --profile-from-start off --launch-count 1 \
  -o $RUN/reports/full_tilelang_int8 \
  $PY $RUN/harness/profile_transpose.py --backend tilelang --dtype int8

$NCU --set source --section SourceCounters \
  --import-source on --profile-from-start off --launch-count 1 \
  -o $RUN/reports/source_tilelang_int8 \
  $PY $RUN/harness/profile_transpose.py --backend tilelang --dtype int8

PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python \
  $PY $RUN/harness/summarize_reports.py
```

Repeat for `fp16` and the `triton` and `cutile` backends.

## Artifacts

- `harness/profile_transpose.py`: exact-config, correctness-checked launcher.
- `harness/summarize_reports.py`: metrics, rules, and opcode extraction.
- `reports/full_*.ncu-rep`: six full plus PM-sampling reports.
- `reports/source_*.ncu-rep`: six source-counter reports.
- `analysis/summary.json` and `summary.tsv`: structured metrics and opcodes.
- `analysis/details_*.txt`: archived NCU details pages.
- `analysis/source_sass_*.txt`: archived source/SASS pages.
- `analysis/stall_hotspots_*.txt`: per-line source-counter attribution.

---

## Independent Verification (2026-07-27)

Re-derived from the same six `.ncu-rep` files plus the full postmerge benchmark
sweep, to check five specific claims. All five hold; one needs a correction.

### 1. INT8 shared-store bank conflicts: 8.6M / 0.255M / 0.011M — CONFIRMED

`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`:

| | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| INT8 store conflicts | **8,587,782** | 254,827 | 10,809 |
| FP16 store conflicts | 703,109 | 12,426 | 188,478 |

Exact match. Note these are the **store** counters specifically; the all-shared
totals are 8.668M / 0.282M / 0.014M.

### 2. Triton and cuTile permute addresses for shared memory — CONFIRMED

The original report proved the resulting *mapping* but explicitly declined to
claim a permutation pass ("they do not prove that either compiler uses an
internal pass literally named 'swizzle'"). The SASS opcode histogram closes
that gap. `PRMT` (byte permute) counts from `analysis/source_sass_*.txt`:

| backend | PRMT (INT8) | PRMT (FP16) | LOP3.LUT (INT8) | INT8 store conflicts |
|---|---:|---:|---:|---:|
| cuTile | **192** | 128 | 77 | 10,809 |
| Triton | **112** | 8 | 48 | 254,827 |
| TileLang | **0** | 60 | 9 | **8,587,782** |

The ordering is monotonic: more permutation, fewer conflicts, across all three
backends. TileLang emits **zero** `PRMT` in its INT8 kernel.

### 3. TileLang permutes for other dtypes but not INT8 — CONFIRMED, with nuance

TileLang emits 60 `PRMT` at FP16 and 0 at INT8, so the claim holds as stated.
But its FP16 permutation is *partial*, not correct-and-sufficient: TileLang
still records 17x Triton's FP16 store conflicts (703K vs 12K) and is still the
slowest of the three at FP16. FP16 survives because of headroom, not because
the layout is right — see claim 5.

### 4. Triton ~1.8x faster than TileLang at INT8 — CONFIRMED (1.86x)

Median over 20 shapes, `results/b200_autotuned_postmerge`:

| dtype | TL/Triton | TL/cuTile | n |
|---|---:|---:|---:|
| fp16 | 1.11 | 1.10 | 20 |
| bf16 | 1.11 | 1.10 | 20 |
| fp32 | **1.02** | **1.02** | 20 |
| int8 | **1.86** | **1.58** | 20 |

The dtype-specificity is unambiguous: FP32 is at parity, FP16/BF16 cost ~10%,
INT8 costs 86%. This is the four-dtype view; the original report covered only
FP16 and INT8.

### 5. "Slower than both about equally" — CORRECTION

True for FP32 (1.02/1.02), FP16 (1.11/1.10) and BF16 (1.11/1.10). **Not true
for INT8**, where the deficit is 1.86x against Triton but 1.58x against cuTile.
The gap is 18% wider against Triton, consistent with the original report's
2.00x/1.68x NCU-side figures.

### Mechanism, restated from the pipe counters

| INT8 | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| **L1TEX throughput (% peak)** | **96.09** | 77.29 | 61.27 |
| **DRAM throughput (% peak)** | **26.57** | 69.85 | 58.46 |
| shared wavefronts | **14,887,274** | 3,644,193 | 3,337,295 |
| store conflicts / shared wavefront | **0.577** | 0.070 | 0.003 |

L1TEX at **96.09% of peak** is a genuinely saturated pipe — a real throughput
bottleneck, not a latency-bound symptom. Conflict replay inflates wavefronts
4.1x, saturates L1, and starves DRAM to 26.6% against Triton's 69.9%. The same
kernel at FP16 sits at 50.4% L1 / 64.7% DRAM, i.e. DRAM-facing with L1 headroom
to absorb its (smaller) conflict count. That headroom is what disappears when
the element width halves.

### Caveat

The `PRMT` counts in claim 2 are **static** SASS counts from the source page,
not dynamic execution counts, so their magnitudes are not directly comparable
to each other or to the dynamic conflict totals. The load-bearing signal is
presence versus absence — 0 vs 112 vs 192 — which is categorical and does not
depend on the static/dynamic distinction.
