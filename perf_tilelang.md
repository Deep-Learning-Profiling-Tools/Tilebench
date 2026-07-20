# TileLang performance notes — B200 / Blackwell

## TMEM allocation gates `tcgen05.mma` (5th-gen tensor cores)

**Status:** root cause confirmed by codegen analysis. The fix is **not yet implemented or
numerically verified.**

### Summary

Every TileLang kernel in this suite allocates its GEMM accumulator with
`T.alloc_fragment` (scope `local.fragment`). On Blackwell that silently disables
`tcgen05.mma` and falls back to Ampere-era per-warp `HMMA`. Triton and cuTile both reach
`tcgen05` on the same problem. This is **not** a TileLang capability gap — the installed
version (0.1.11) implements `tcgen05` fully — it is a consequence of TileLang deriving
instruction selection from author-declared memory scopes.

### Measured impact (B200, `results/runs/20260707_194904_...`, matmul_fp32_fp16_fp8)

Times in ms, M=N=4096, smallest K case:

| dtype | torch | Triton | cuTile | TileLang | TL vs best |
|---|---|---|---|---|---|
| fp32 | 0.562 | 0.344 | 0.072 | 0.169 | 2.3× (TL beats Triton and torch) |
| fp16 | 0.027 | 0.043 | 0.044 | 0.092 | 3.4× |
| fp8_e4m3fn | 0.604 | 0.034 | 0.023 | 0.129 | **5.7×** |

The fp8 gap **widens with K**, from 5.7× at the smallest K to **9.7×** at the largest —
the signature of a per-element conversion cost scaling with the reduction dimension.

fp32 is TileLang's best dtype precisely because there are no fp8/fp16 tensor cores to
lose there.

### Codegen evidence (sm_100a, offline)

Same problem, same 128×128 tile:

| dtype | backend | tensor-core op | count | fp8 unpack |
|---|---|---|---|---|
| fp16 | Triton | `UTCHMMA` (tcgen05) | 8 | — |
| fp16 | cuTile | `UTCHMMA` (tcgen05) | 4 | — |
| fp16 | **TileLang** | `HMMA.16816.F32` | **256** | — |
| fp8 | Triton | `UTCQMMA` (tcgen05, native fp8) | 4 | 0 |
| fp8 | cuTile | `UTCQMMA` (tcgen05, native fp8) | 2 | 0 |
| fp8 | **TileLang** | `HMMA.16816.F32` (fp16 math) | **64** | **64** |

`UTC*` is the tcgen05 instruction family. One `UTCQMMA` consumes a whole tile against
tensor memory; TileLang issues 64 per-warp 16×8×16 MMAs for the same work.

At fp8, `HMMA` has no fp8 mode, so TileLang must upcast every operand:

```
tilelang:  64 LDS.U8 + 64 F2FP.F16.E4M3.UNPACK_B + 128 FADD + 64 HMMA.16816.F32
triton  :  4 UTCQMMA
cutile  :  2 UTCQMMA
```

All three emit 64 `F2FP.SATFINITE.E4M3...PACK` — that is the fp8 *output* store, common
to everyone. TileLang's extra cost is the **input unpack**.

Note TileLang *does* use TMA (`UTMALDG` ×34) for data movement. The deficit is purely in
the compute path: modern memory path, legacy tensor cores.

### Root cause

`src/cuda/op/gemm.cc`, `SelectInst`:

```cpp
if (AllowTcgen5Mma(op, target)) return kCudaTCGEN05;
if (AllowWgmma(op, block_size, target)) return kCudaWGMMA;   // requires Hopper
return kCudaMMA;
```

`AllowWgmma` requires `TargetIsHopper`, so on sm_100 it is dead. Selection therefore
reduces to `AllowTcgen5Mma`:

```cpp
bool scope_ok = (IsSharedBuffer(op.a_) || op.a_.scope() == "shared.tmem") &&
                IsSharedBuffer(op.b_) && op.c_.scope() == "shared.tmem";
if (!TargetIsSm100(target) || !scope_ok) return false;
return GetTCGEN5MMAMeta(op.m_, op.n_, op.k_, ab_dtype, op.c_->dtype).first;
```

- **A** — shared, or optionally tmem
- **B** — shared only
- **C (accumulator)** — **must** be `shared.tmem`

Our kernels use `a_tile`/`b_tile` = `T.alloc_shared` (pass) and:

```python
acc = T.alloc_fragment((BLOCK_SIZE_M, BLOCK_SIZE_N), "float32")   # local.fragment — FAILS
```

So on Blackwell, instruction selection is a **pure binary keyed on the accumulator's
scope**. There is no compiler decision:

- `acc` in fragment → `kCudaMMA`, always
- `acc` in tmem → `kCudaTCGEN05`, always

**Confirmed empirically:** a full 32-config sweep of `matmul_fp32_fp16_fp8` at
`sm_100a` (every tile shape, thread count, stage count in the autotune space) gives
`tcgen05 = 0` and `tmem = 0` in **32/32** configs. No config reaches it. Autotuning
cannot fix this.

### The shapes already qualify

`src/op/tcgen5_meta.h`, `GetTCGEN5MMAMeta`:

- fp16/bf16 → fp32: `M % 128 == 0`, `K % 16 == 0`, N divisible by an atom
- fp8 → fp32: `M % 128 == 0`, `K % 32 == 0`, N divisible by an atom

Our 128×128×128 tile passes both. The hardware path is reachable at the exact shapes
already being benchmarked.

### Secondary finding: register spills

20 of 32 configs spill, up to **1593** `LDL`/`STL` instructions. Every `threads=128`
config with BM or BN = 256 spills badly. A 128×128 fp32 accumulator is 16K floats in the
register file. Per `T.alloc_tmem`'s docstring, TMEM is *"designed to reduce register
pressure"* — so moving the accumulator to tensor memory should address this too.

### Affected operators

Every TileLang impl using `T.gemm` with an `alloc_fragment` accumulator:

- `2d_conv`
- `batched_matmul`
- `block_sparse_attention`
- `flash_attention`
- `matmul_fp32_fp16_fp8`
- `matmul_int8`
- `streamk_matmul`

**Zero** of the 45 TileLang impls currently use `T.alloc_tmem`. The remaining 38
operators do not use tensor cores and are unaffected by this issue.

### Why Triton and cuTile are unaffected

Neither declares tensor memory — or any memory space — at all:

```python
# impl_triton.py:66,73
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
accumulator = tl.dot(a, b, accumulator)

# impl_cutile.py:76,88
acc = ct.zeros((TM, TN), dtype=ct.float32)
acc = ct.mma(a_tile, b_tile, acc)
```

Their compilers see `sm_100` and route the accumulator to TMEM themselves.

| | memory scope | on new hardware |
|---|---|---|
| Triton, cuTile | compiler-inferred | same source auto-upgrades to `tcgen05` |
| TileLang | author-declared | source is implicitly pinned to a GPU generation |

This has a methodological consequence for the comparison: for these 7 operators, Triton
and cuTile numbers reflect *compiler* instruction selection, while TileLang numbers
reflect the *author's* choice. Those are different quantities and should not be
presented on a single "which DSL is faster" axis without stating this.

### The fix

```python
acc = T.alloc_tmem((BLOCK_SIZE_M, BLOCK_SIZE_N), "float32")
```

`T.alloc_tmem` (`tilelang/language/allocate.py:228`) is the **only** thing in the entire
package that creates a `shared.tmem` buffer. There is no promotion pass and no config
knob — `lower_shared_tmem.cc` only *consumes* existing tmem buffers.

Constraints from the docstring:

- TMEM is 128 lanes × 512 columns × 32 bits
- Column count must be a power of 2, ≥ 32, ≤ 512
- Allocation and deallocation must be in the same warp
- Auto-deallocated at end of block; `T.deallocate_tmem` only for earlier release
- "All pre-processing must occur before data is loaded into TMEM, and all post-processing
  after data is retrieved"

The 128-lane limit caps **BLOCK_SIZE_M at 128**; our search spaces include
`BLOCK_SIZE_M=256`, which would need the 2-CTA path (`use_2cta` in `GetTCGEN5MMAMeta`).

Not yet verified: whether `T.clear(acc)` and the epilogue `T.copy(acc, c[...])` work
directly against a TMEM buffer or need staging.

### Recommended safety net

`T.tcgen05_gemm()` (`tilelang/language/gemm_op.py:236`) sets `is_tcgen05: 1` and runs the
**same** scope check, but on failure calls `LOG(FATAL)` with the offending scopes and
shapes instead of silently degrading:

```cpp
if (op.isTcgen05_) {
  if (!AllowTcgen5Mma(op, target)) FatalTcgen5Unavailable(op, target);
  return kCudaTCGEN05;
}
```

It does **not** auto-allocate TMEM — `alloc_tmem` is still required. Its only value is
turning a silent 5.7–9.7× regression into a build error. The silence is arguably the
real bug here.

### Reproduction

```bash
PYTHONPATH=. python sweep_configs_sm100.py --operator matmul_fp32_fp16_fp8 --dtype fp8_e4m3fn
PYTHONPATH=. python sweep_configs_sm100.py --operator matmul_fp32_fp16_fp8 --dtype fp16 --limit 2
```

Artifacts: `results/sweeps_sm100a/matmul_fp32_fp16_fp8/{fp16,fp8_e4m3fn}/sweep.json`.

Requires `CUDA_HOME` pointing at a Blackwell-capable toolkit; no B200 needed, since
codegen is host-side.

### Caveats

- All of the above is **codegen analysis, not execution.** The B200 timings are real, but
  the causal link runs through instruction counts, not a controlled experiment.
- The fix has not been implemented, compiled, or checked for numerical correctness.
- `sweep_configs_sm100.py` counts opcodes, so it is **blind to immediate-value changes**.
  Example: `GROUP_SIZE_M` 64 vs 128 yields identical opcode counts but different swizzle
  immediates (`IMAD.SHL.U32 R8, R12, 0x40` vs `0x80`) and therefore different L2
  behaviour. Do not read "identical feature row" as "identical kernel."
- Instruction counts compare kernels at one tile config; cross-config comparisons of
  `insns` are confounded by tile size.

---

## Histogramming — the same mechanism, opposite sign

TileLang's **largest win** in the suite has the same root variable as its largest loss:
which memory space the accumulator lives in.

### Measured (B200, same run)

| operator | TileLang / best competitor |
|---|---|
| histogramming | **0.33×** (3× *faster*) |
| 2d_max_pooling | 0.72× |
| mean_reduction | 0.83× |
| moe_topk_gating | 0.86× |

### Same algorithm, one difference

All three backends implement the identical two-stage privatized histogram
(per-program partial histograms, then a reduce kernel). The only difference is **where
the partial histogram is accumulated**:

| backend | stage-1 accumulation | memory space |
|---|---|---|
| **TileLang** | `T.atomic_add(smem[val], 1)` then `T.copy(smem, partial[pid, :])` | **shared** |
| Triton | `tl.atomic_add(row_base + safe_bins * stride_pb, one, mask=valid)` | **global** |
| cuTile | `ct.atomic_add(partial_ptr, (row_idx, bin_idx), update)` | **global** |

TileLang declares `smem = T.alloc_shared((num_bins,), "int32")` and flushes once per
block. Triton and cuTile hit global memory on every element. Shared-memory atomics are
substantially cheaper and do not contend across SMs — that is the 3×.

Triton and cuTile are algorithmically equivalent to each other here; neither is doing
anything naive. Both privatize. They just privatize into global memory.

### What each DSL's best case would be

**Triton — `tl.histogram` would be better and is not being used.** It exists as a
first-class compiler intrinsic (`tt.histogram`, `language/semantic.py:1794`, lowered by
`HistogramOpToLLVM.cpp`) and returns a register-resident tile of shape `[num_bins]`,
avoiding global atomics entirely. The committed `impl_triton.py` does not use it, so the
current Triton number is **not** Triton's best case. *(Verified: the intrinsic exists and
avoids global atomics. Not verified: whether its internal lowering uses shared memory,
per-thread registers with ballot/popcount, or a mix — that lives in C++ we did not
inspect.)*

**cuTile — appears unable to express this at all.** Its 132-name public API contains no
shared-memory or scratch allocation primitive (no `alloc_shared`, `smem`, `scratch`, and
no histogram intrinsic). Both `ct.atomic_add` and `ct.scatter` are documented as operating
on an `array` — a global-memory kernel parameter — not on a tile. Tiles are cuTile's
register/value abstraction and are compiler-placed, with no author-facing way to perform
scattered atomic accumulation into a shared buffer. *(Evidence is strong but not
exhaustive: based on public API enumeration and docstrings, not on cuTile's internals or
documentation.)*

So the honest ranking of *best achievable* here is likely
TileLang ≈ Triton-with-`tl.histogram` > cuTile, whereas the *measured* ranking is
TileLang > Triton ≈ cuTile. The Triton gap is an authoring choice; the cuTile gap looks
structural.

### Why this matters for the thesis

The same property — TileLang's explicit, author-declared memory scopes — produces both
extremes in the suite:

| | accumulator scope | result |
|---|---|---|
| matmul (fp8) | `local.fragment` instead of TMEM | **5.7–9.7× slower** |
| histogram | `shared` instead of global | **3× faster** |

Explicit memory control is not simply a liability. It rewards an author who knows the
hardware and penalizes one who does not — and in both directions the compiler is silent.
Triton and cuTile trade that ceiling for a higher floor: their compilers make the choice,
so they auto-upgrade on new hardware (matmul → `tcgen05`) but cannot be hand-steered when
the author knows better (histogram → shared memory).

This is a cleaner controlled comparison than matmul, because all three implementations
share the same algorithm and two-kernel structure, with memory space as the single
varying factor.

### Open question

Whether Triton and cuTile *could* privatize into shared memory determines whether this is
an authoring gap or a DSL limitation — the distinction matters for the writeup. Current
evidence: Triton yes (via `tl.histogram`), cuTile apparently no. Confirming the cuTile
side against its official documentation is worth doing before publishing that claim.

---

## top_k_selection — integer division lowering (a compiler-optimization gap)

**Status:** root cause confirmed three independent ways — SASS statics, a controlled
timing ablation, and NCU hardware counters (RTX 4060). The shift/mask fix is written and
numerically verified but not committed.

This finding is a **different mechanism** from the TMEM issue above, and the distinction
matters for the thesis. TMEM is a *language-design* consequence (author-declared scopes).
This one is a plain *compiler-optimization* gap: Triton's compiler strength-reduces and
CSEs integer division; TileLang's lowering does neither. Same source-level semantics in,
~9× the hardware cost out. It is a straightforward Triton win.

### Measured (B200, same run) — and what the headline number actually was

All 20 cases are fp32; 5 N × 4 k:

| N | torch | Triton | cuTile | TileLang | TL/Triton | TL/best |
|---|---|---|---|---|---|---|
| 4096 | 0.024 | 0.134 | 0.182 | 0.556 | 4.16 | 23.5 |
| 16384 | 0.083 | 0.176 | 0.245 | 0.755 | 4.28 | 9.1 |
| 65536 | 0.050 | 0.254 | 0.339 | 1.003 | 3.95 | 20.0 |
| 262144 | 0.077 | 0.326 | 0.432 | 1.267 | 3.88 | 16.5 |
| 1048576 | 0.155 | 0.568 | 0.778 | 1.950 | 3.43 | 12.6 |

Two things the raw ranking hides:

- **The "15–23× slower" figure is TileLang vs torch, not vs a DSL.** `torch.topk` is
  radix-select (O(N)); all three DSL impls run a full bitonic sort (O(N log²N)). torch
  beats *every* DSL by 2–20×. That is an algorithm gap shared by all three impls, not a
  codegen fact about any DSL. DSL-vs-DSL comparisons for this op should exclude torch or
  state the algorithmic difference explicitly.
- **The real TileLang-vs-Triton gap is a flat ~3.4–4.3×**, essentially constant across
  all N and k (k has no effect on any backend — the sort dominates).

### Root cause

The kernel computes, per element, `offset // stride`, `offset % stride`, and
`(slice_1_offset // stage) % 2` with **runtime** `stride`/`stage` (kernel arguments; the
launch loop halves `stride` each step). Two lowering failures compound:

1. **Int64 promotion with a guarded slow path.** TVM promotes the index arithmetic to
   int64 when it cannot prove 32-bit safety. Each division site emits
   `cvt.u64.u32` + branch + fast `div.u32` path + full `div.s64` fallback. The fast path
   is taken at runtime, but every site pays the guard and the code bloat.
2. **No CSE of the divisor reciprocal.** GPU integer division lowers to
   `MUFU.RCP`-based emulation: one reciprocal of the divisor, then a multiply+fixup per
   numerator. The divisor is *uniform across the whole kernel*. Triton computes the
   reciprocal once; TileLang materializes an independent full sequence per unrolled
   element (`T.Parallel` unrolls 8 elements/thread, each re-deriving everything).

Statics at sm_100a (`results/sweeps_sm100a/top_k_selection/`):

| | TileLang | Triton |
|---|---|---|
| PTX divisions | **56** (24 `div.s32`, 16 `div.u32`, 16 `div.s64`) | **8** (`div.s32`) |
| SASS `MUFU.RCP` (≈ distinct division expansions) | **41** | **1** |
| total SASS instructions | 2432 | 264 |

### Ablation (RTX 4060, single compare-exchange launch, N=2^20)

Four explicitly separate kernels, outputs bit-identical to the original, interleaved
round-robin timing (clock-drift-proof), median of 10 rounds:

| variant | stage=1024, stride=512 | stage=2^20, stride=2^19 |
|---|---|---|
| A: `//`,`%` everywhere (current impl) | 29.9 µs (1.00×) | 29.7 µs (1.00×) |
| B: shift address, div direction | 17.6 µs (1.70×) | 16.6 µs (1.79×) |
| C: div address, shift direction | 22.9 µs (1.31×) | 20.6 µs (1.44×) |
| D: shift/mask everywhere | **12.8 µs (2.34×)** | **11.4 µs (2.61×)** |

A 20-point scan across the full sort schedule (every `stride` from 1 to 2^19) gives
div/shift = 2.0–2.4× at every stride, 2.05× aggregate. The two sites compose
multiplicatively; the address computation is the bigger one.

The rewrite is **algorithmically identical**, not a different algorithm: `stride` and
`stage` are powers of two by construction (`_next_pow2` padding), so
`x // stride == x >> log_stride` and `x % stride == x & (stride - 1)` exactly. Verified
against `torch.topk` at N = 4096 / 65536 / 262144.

### NCU profiles (RTX 4060, same mid-sort launch, N=2^20, grid 512×128)

| | TileLang div (current) | TileLang shift | Triton |
|---|---|---|---|
| duration | 41.7 µs | 18.6 µs | 19.5 µs |
| DRAM throughput | 40% | **86%** | **82%** |
| Compute (SM) throughput | **62%** | 48% | 22% |
| XU pipe (`MUFU.RCP` = idiv emulation) | **28.3%** | 0% | 3.2% |
| ALU pipe | 65.5% (NCU-flagged bottleneck) | 52.5% | 23.6% |
| executed instructions | 4,345,856 | 1,128,448 | 679,936 |
| registers/thread | 43 | 28 | 37 |
| achieved occupancy | 68.9% | 88.3% | 80.4% |
| dominant stalls | wait 3.0 + math_throttle 1.8 + long_sb 3.3 /issue | long_scoreboard 22.1 | long_scoreboard 13.4 + lg_throttle 11.8 |

Reading:

- The div kernel is **compute-bound on exactly the division pipes**: XU has no other
  possible source in a sort kernel (no transcendentals), and the `I2F`/`F2I` conversions
  in the mix exist only because idiv emulation round-trips through floating point. DRAM
  sits 60% idle while the SM grinds 3.85× more instructions.
- The shift kernel is at the **memory roofline** (86% DRAM, stalls almost purely
  long_scoreboard) — the profile a compare-exchange pass *should* have. So a bitonic step
  is **not** inherently compute-bound; it is compute-bound only because of this lowering.
- Stall bars are per-issued-instruction ratios; denominated by instruction count, both
  variants pay ~the same memory-wait warp-cycles (the irreducible data movement) and the
  div variant adds **~26M warp-cycles of pure compute/issue overhead** on top (57M vs
  31M total). That surplus is the whole difference.
- **TileLang-with-shifts matches Triton per launch (18.6 vs 19.5 µs).** For this kernel
  the entire per-launch gap is the division lowering.
- Triton's XU = 3.2%, not 0: it divides too — **once** (ptxas hoists the uniform divisor
  reciprocal; 1 `MUFU.RCP` vs TileLang's 41).

### Hypotheses killed along the way

- **`T.if_then_else` → `T.Select`: no effect (~1.0×).** An earlier 4.9× claim was a
  benchmarking artifact (sequential variants mutating a shared in-place buffer). The
  `T.Select` form is kept in the impl anyway (it is not worse, and is vectorization-safe).
- **Branchy guarded stores: not a cost here.** `LegalizeSafeMemoryAccess` lowers guarded
  stores to real branches (statement-level `IfThenElse`), but every benchmark case has
  N equal to the padded power of two, so the guards are uniformly true: NCU shows 100%
  branch efficiency, 0 divergent branches, in both variants.
- **Small-N cases measure launch overhead, not the kernel.** N=4096 launches **2 blocks
  × 128 threads** — under 2% utilization on a 24-SM card (and far less on a B200). At
  small N all DSL timings are dominated by launch latency and serial dependency chains;
  throughput analysis is only meaningful at N ≥ ~2^19.

### The fix

In `bitonic_step_kernel`, pass `log_stage`/`log_stride` alongside (or instead of)
`stage`/`stride` and replace:

```python
slice_1_offset = (offset // stride) * (2 * stride) + (offset % stride)
descend = ((slice_1_offset // stage) % 2) == 1
```

with:

```python
slice_1_offset = ((offset >> log_stride) << (log_stride + 1)) | (offset & (stride - 1))
descend = ((slice_1_offset >> log_stage) & 1) == 1
```

with the launch loop supplying `stage.bit_length() - 1` / `stride.bit_length() - 1`.
This is what `impl_tilelang.py` for **bitonic_sort** already does by hand — and that op
measures ~1.02× vs Triton, consistent with everything above.

### Reproduction

```bash
# statics
PYTHONPATH=. python sweep_configs_sm100.py --operator top_k_selection --dump

# NCU (needs GPU perf-counter permission; on WSL2 enable it in the Windows
# NVIDIA Control Panel -> Developer -> Manage GPU Performance Counters)
ncu --set full --kernel-name main_kernel --launch-skip 100 --launch-count 1 \
    -f -o tl_topk \
    python -c "import sys, torch; sys.path.insert(0, 'benchmarks/operators/top_k_selection'); import impl_tilelang as m; x = torch.randn(1<<20, device='cuda'); m.run(x, 1<<20, 16)"
```

Note: NCU cannot wrap `scripts/run_bench.py` — the harness times via Proton, and CUPTI
allows one profiler per process (`cuptiSubscribe` error 39). Profile the impl directly.

### Caveats

- Ablation and NCU numbers are **RTX 4060 (sm_89), not B200**. The mechanism
  (instruction volume, XU traffic, regime flip) is architecture-independent, and the
  ~2.3–2.6× measured explains most of the flat ~3.4–4.3× B200 gap, but the exact B200
  split is unverified — the same NCU commands run unchanged there.
- Single mid-sort launch, steady-state buffer. Division cost is data-independent, so
  this is safe for the div-vs-shift comparison, but absolute times vary with sort phase.
- One trap worth recording: a factory-generated variant with a Python-level `if` inside
  `@T.prim_func` was **not** pruned at trace time — the "shift" variants silently kept
  the division code, making all variants time identically. Variant kernels must be
  written as separate top-level `prim_func`s (verify via div-token counts in
  `get_kernel_source()`).
