# batched_matmul — why TileLang is slower on B200

fp16, BATCH=32, M=N=K=640 (largest case in the grid). Each backend at the tile
its own autotuner recorded in
`results/b200_autotuned_postmerge/operators/batched_matmul.json`.
B200 sm_100, 148 SMs, NCU 2026.1.1, pinned to an idle GPU by UUID.
Method: `kernel-perf-differential` skill.

## Verdict

TileLang is slower than **both** Triton and cuTile in **17 of 20** fp16 cases
(median 1.19x / 1.26x). At M=640 it is 1.39x Triton and 1.36x cuTile.

**Established:** for identical MMA work, TileLang executes **2.44x cuTile's
instructions**, and the excess is per-thread address arithmetic for `cp.async`
tile loads. TileLang's BMM kernel does not use TMA, because TMA lives behind
warp specialization and this operator disables it.

**Not established:** that removing that arithmetic fixes the gap. Enabling warp
specialization *did* switch the kernel onto TMA and produced **no speedup**. So
the instruction excess is real and its origin is identified, but its causal
contribution to the runtime is unproven — and the obvious fix does not work.

## Measurements

| | TileLang | Triton | cuTile | torch |
|---|---:|---:|---:|---:|
| duration (ns) | **41,568** | 29,856 | 30,464 | **21,472** |
| `sm__cycles_active.avg` | 58,337 | 40,617 | 43,509 | 25,230 |
| tensor % of peak | **23.72** | 34.07 | 31.80 | **54.85** |
| inst executed / SM | 25,286 | 33,351 | **10,356** | 9,493 |
| inst / cycle | 0.433 | **0.821** | 0.238 | 0.376 |
| active warps / sched | **1.008** | 2.517 | **4.527** | 1.567 |
| eligible warps / sched | 0.112 | 0.235 | 0.066 | 0.100 |

## Step 1 — configuration is matched

| backend | tile | rest |
|---|---|---|
| TileLang | 128x128x**64** | threads=128, stages=4 |
| Triton | 128x128x**32** | warps=4, stages=4, group=8 |
| cuTile | 128x128x**64** | occupancy=4, group=1 |

All three converge on a 128x128 output tile. **TileLang and cuTile are identical
including BK**, so that pair has no configuration degrees of freedom left.
Triton's and cuTile's `_DEFAULT_CONFIG` were verified equal to their M=640
autotune winners, so the capture is genuinely tuned-vs-tuned.

## Step 2 — correctness

TileLang `max_abs_err = 0.0001` vs an fp32 `torch.bmm` reference
(`atol=1.0, rtol=1e-2`). Nothing is skipping work.

## Step 3 — essential work is identical

```
tensor-busy cycles/SM = pipe_tensor_cycles_active.pct_of_peak_active/100 * sm__cycles_active.avg

  tilelang 13,838    triton 13,838    cutile 13,838    torch 13,838
```

Identical to the cycle. **100% of the gap is overhead.**

## The cycle budget

`cycles = instructions / issue_rate` closes for every backend:

| vs Triton | instr ratio | issue-rate ratio | predicted cycles | measured |
|---|---:|---:|---:|---:|
| TileLang | 0.76x | 0.54x | 1.40x | **1.44x** |
| cuTile | 0.31x | 0.29x | 1.06x | **1.07x** |
| torch | 0.28x | 0.46x | 0.62x | **0.62x** |

TileLang loses on **two different axes** depending on the comparison:

- **vs cuTile** (identical tile) — 2.44x the instructions. cuTile issues
  *slower* than TileLang (0.238 vs 0.433) and still wins, purely by having less
  to execute.
- **vs Triton** — *fewer* instructions (0.76x) but half the issue rate, because
  TileLang runs 1.0 warp/scheduler against Triton's 2.5.

There is no single bottleneck. TileLang emits more code than cuTile *and* has
fewer warps than Triton.

## Where the instructions come from

The generated CUDA is in
`~/.tilelang/cache/0.1.11-x86_64/kernels/cf0fd59d3bad*/device_kernel.cu`.

```
cp.async.bulk.tensor      0      <- TMA absent
tensormap                 0
cp_async_gs              20      <- Ampere-generation async copy
tcgen05                  40      <- the MMA itself IS Blackwell
```

The kernel is a hybrid: **Blackwell `tcgen05` MMA fed by Ampere-era `cp.async`
loads.** Each load carries a fully expanded per-thread address expression —
**~46 integer ops** (14 adds, 13 multiplies, 12 ands, 7 shifts):

```c
tl::cp_async_gs<16>(&((half_t*)A_tile)[
    (k * 8192) + (i_8 * 2048) + ((threadIdx.x >> 3) * 64)
  + (((((threadIdx.x & 63) >> 5) + ((threadIdx.x & 7) >> 2)) & 1) * 32)
  + (((((threadIdx.x & 31) >> 4) + ((threadIdx.x & 3) >> 1)) & 1) * 16)
  + (((((threadIdx.x & 15) >> 3) +  (threadIdx.x & 1))       & 1) *  8)],
  &A[ ... blockIdx.z * 262144 + blockIdx.x * 65536 + ... ]);
```

The three bracketed terms are the **XOR swizzle, recomputed from `threadIdx.x`
on every single load**, in the vector datapath, by all 128 threads. There are
**8 such loads per K-iteration** (4 for A, 4 for B).

That is what TMA exists to remove: one descriptor built on the host, swizzle
encoded in it, hardware copy engine walks the tile, zero per-thread addressing.
It accounts for the pipe mix against cuTile:

| pipe | TileLang | cuTile | ratio |
|---|---:|---:|---:|
| `lsu` | 8.358% | 1.454% | **5.7x** |
| `fma` | 2.650% | 0.397% | **6.7x** |
| `alu` | 6.741% | 3.820% | 1.8x |
| `adu` | 4.642% | 6.076% | 0.76x |

Two further structural costs in the same loop:

- **Only 1 of 4 warps issues the MMA** — `if ((threadIdx.x >> 5) == 0)` guards
  descriptor setup and all four `tcgen05mma_ss` calls. The other three warps only
  `mbar[0].wait(0)`. 75% of the CTA is idle during the math.
- **Four sync points per K-iteration**: `cp_async_wait<3>`, `__syncthreads()`,
  `mbar.wait(0)`, `__syncthreads()`, `__syncthreads()`.

## Why there is no TMA: it is gated behind warp specialization

```
matmul_fp32_fp16_fp8   TL_DISABLE_WARP_SPECIALIZED: False  -> cp.async.bulk.tensor: 6  (384 thr)
batched_matmul         TL_DISABLE_WARP_SPECIALIZED: True   -> cp.async.bulk.tensor: 0  (128 thr)
streamk_matmul         TL_DISABLE_WARP_SPECIALIZED: True
```

Confirmed across cached kernels: every warp-specialized TileLang kernel emits
`cp.async.bulk.tensor` and runs 256-384 threads; every non-specialized one emits
`cp_async_gs` and runs 128-256. Same compiler, same hardware. TileLang has TMA;
this operator opts out of it.

This is also the mechanism behind the +13-15% that flipping the same flag was
worth on `matmul_fp32_fp16_fp8` — it was not merely better scheduling, it
switched the load path onto TMA.

## The intervention that should have worked, and didn't

Setting `TL_DISABLE_WARP_SPECIALIZED: False` on batched_matmul:

- **The kernel does switch to TMA.** Verified in the compiled kernels:
  `cp.async.bulk.tensor = 3`, `cp_async_gs = 0`, `threads = 256`.
- **Output stays correct.**

| M | dtype | max_abs_err | verdict | ms |
|---|---|---:|---|---:|
| 128 | fp16 | 0.00006 | PASS | 0.0046 |
| 128 | bf16 | 0.00048 | PASS | 0.0046 |
| 640 | fp16 | 0.00012 | PASS | 0.0387 |
| 640 | bf16 | 0.00098 | PASS | 0.0384 |

- **It is not faster.** 0.0387 ms with TMA vs 0.0374 ms without, at M=640 — a
  hair slower, and still ~1.3x Triton and cuTile.

Removing the address arithmetic did not close the gap. Any explanation resting
on that arithmetic must account for this.

**Leading hypothesis, untested: the K-loop is too short to amortize.** BMM at
M=640 runs **10 K-iterations** (640/64); `matmul_fp32_fp16_fp8` runs **128**
(8192/64). TMA's benefit is per-iteration steady state, while warp
specialization adds fixed prologue cost — producer/consumer split, mbarrier
setup, 2x the threads. Over 128 iterations that trades well (+13-15%); over 10
it may not pay for itself. The whole BMM grid has this property, which would
explain losing in 17 of 20 cases.

Testable: sweep K with tile fixed, WS on vs off, and check whether the TMA
version overtakes at large K.

## What is eliminated

| hypothesis | killed by |
|---|---|
| bad tile choice | TileLang and cuTile autotune to identical 128x128x64 |
| doing more math | 13,838 tensor cycles, identical across all four backends |
| tile / wave quantization | corr −0.12 and −0.09 with the gap; the 16x-wasted M=32 case has the *smallest* gap |
| warp starvation | TileLang has **more** eligible warps than cuTile and torch, both faster |
| low occupancy | raising it (threads 128 -> 256) made it **slower**; torch also runs 1 CTA/SM and is fastest |
| stall composition | every backend is 92-101% stalled; cuTile is the most memory-stalled and beats TileLang |
| missing uniform datapath | cuTile is also near-zero (0.727%) with 2.44x **fewer** instructions; Triton has the most uniform *and* the most instructions |
| no TMA | enabling it changed nothing (above) |

### On the uniform-datapath idea specifically

It was carried over from the matmul report and does not survive here:

| | uniform % | inst/SM | duration ns |
|---|---:|---:|---:|
| Triton | 14.433 | 33,351 | 29,856 |
| torch | 5.368 | 9,493 | 21,472 |
| cuTile | 0.727 | 10,356 | 30,464 |
| TileLang | 0.083 | 25,286 | 41,568 |

No relationship between uniform usage and instruction count. Triton has 20x
cuTile's uniform usage and more instructions than anyone.

## Stall breakdown (normalized by resident warps)

Raw warp-cycles are meaningless here — resident warps range 1.0 to 4.5.

| stall | TileLang | Triton | cuTile | torch |
|---|---:|---:|---:|---:|
| `long_scoreboard` | 46.1% | 55.5% | **90.5%** | 65.5% |
| `barrier` | 18.0% | 18.2% | 1.6% | 2.5% |
| `wait` (arith dep) | 14.8% | 11.5% | 4.2% | 7.9% |
| **TOTAL stalled** | **92.1%** | **92.5%** | **101.0%** | **94.1%** |

## Configuration crashes

Pinning tile params uses TileLang's **direct-JIT** path, which has no timeout or
exception net — unlike the autotuner, which wraps every candidate in
`run_with_timeout()` and silently discards failures
(`tilelang/autotuner/tuner.py` lines 47-163, 585, 621, 664, 688). Configs the
autotuner quietly drops crash the process here. This is why autotune "works"
while the default benchmark path errors.

| tile | stages | smem | result |
|---|---:|---:|---|
| 128x128x64 | 4 | 131,072 B | OK |
| 128x128x64 | 3 | 98,304 B | OK |
| 128x128x64 | 2 | 65,536 B | **crash** |
| 128x128x32 | 2 | 32,768 B | **crash** |
| 64x64x64 | 4 | 65,536 B | **crash** |
| 64x64x32 | 2 | 16,384 B | **crash** |
| 32x32x64 | 2 | — | **crash** (AssertionError) |

Two independent failure modes: `BLOCK_M=64` fails at a stage count that works at
128, and `num_stages=2` fails at a tile that works at stages 3 and 4.
`_DEFAULT_CONFIG` is `64/64/32, num_stages=2` with `config.yaml` setting
`autotune: false`, so the **default path runs a config that hits both**.

## Tooling notes

- **NCU hangs on TileLang BMM kernels.** `--set source --section SourceCounters`
  hung at every shape (M=640, 384, 128), on idle and contended GPUs, with and
  without `--replay-mode range`. The plain counter capture then also hung on the
  warp-specialized variant. The same captures succeed on
  `matmul_fp32_fp16_fp8`. Unexplained.
- **The profiler was not needed for the decisive finding.** `device_kernel.cu`
  in TileLang's kernel cache answers "does it emit TMA" categorically, in one
  grep. Reach for generated source before a profiler when the question is
  presence/absence.
- `executable.so` in the kernel cache is host-only (`cuobjdump: does not contain
  device code`), so that disassembly route is closed.
- GPU 0 was found running another user's job at 100% util plus a stale harness
  process of ours; everything was re-captured on an idle GPU. The difference was
  +0.1% to −2.3%, so both agree, but the clean numbers are reported.

## Next steps, in order of value

1. **K-sweep with WS on vs off**, tile fixed — tests the amortization
   hypothesis, and is the only thing that would explain the failed TMA
   intervention.
2. **Get the pipe mix for the WS/TMA variant** (NCU hung). Without it we do not
   know whether the instruction count actually dropped when TMA turned on.
3. **Fix the `num_stages=2` crash** — it blocks the clean occupancy test.
4. **Re-autotune with WS enabled**; the tile/stage/thread optimum shifts once
   the load path changes, so the current winner is tuned for the wrong codegen.

## Reproducing

```bash
source harness/env.sh
bash harness/run_ncu.sh               # 4 backends x 6 sections -> reports/c_*
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python:$PYTHONPATH \
  $PY ~/.claude/skills/kernel-perf-differential/helpers/scheduler_budget.py \
    reports/c_tilelang_fp16_M640.ncu-rep:tilelang \
    reports/c_cutile_fp16_M640.ncu-rep:cutile \
    reports/c_triton_fp16_M640.ncu-rep:triton

# the decisive check needs no profiler:
grep -c "cp.async.bulk.tensor" ~/.tilelang/cache/*/kernels/<hash>/device_kernel.cu
```
