# Revision map — `skills/reference/triton/3.6.0/SKILL.md`

Source: `skills/triton-guide/SKILL.md` (sha256 `ac48c68a4d980979ac5e8d90d7ce97e2db6ff97f7af0642ea659564920f8d863`, 1644 lines, Triton 3.6.0). Section numbers, headings and line ranges are those of the source; classification follows `docs/llm_v2/skill_inventory.json`. "New section" refers to the revised document.

Actions: **kept** (content preserved, wording may be tightened) · **kept-edited** (content corrected or trimmed against the installed 3.6.0 package) · **moved-to-device-context** (hardware facts handed to the Device Context author; not in the reference) · **removed** (out of scope for a Reference Skill).

## Section table

| Source section (lines) | Action | New section | Reason |
|---|---|---|---|
| frontmatter (1–7) | removed | frontmatter | replaced by the v2 frontmatter (`name: triton-reference`, `dsl`, `version`, `kind`, `derived_from`, `revised`); old `user-invocable`, `allowed-tools`, `argument-hint` are Claude Code loader fields |
| title + intro (9–15) | kept-edited | title | `lib/Triton/triton-3.6.0/` path removed (absent from the worktree); scope statement added; one line that `@triton.autotune`/`Config`/`do_bench` are not permitted in this setting |
| §1 Import Boilerplate (17–48) | kept-edited | 1 | `triton.profiler as proton` row dropped (TileBench timer concern); `triton.testing` row dropped (benchmarking); `autotune`, `Config`, `Autotuner`, `Heuristics` removed from the namespace table; `libdevice` import added (verified) |
| §2.1 DType Constants (52–68) | kept-edited | 2.1 | fp8 table re-labelled; capability-dependent availability added from `backends/nvidia/compiler.py::parse_options` (`fp8e4nv` needs cap >= 8.9; `fp8e4b15` deprecated on >= 9.0; AMD formats up-cast); dtype predicate list expanded from `dir(tl.dtype)` |
| §2.2 Dtype of a Pointer / Tensor (70–77) | kept | 2.2 | — |
| §2.3 Common Casts (79–92) | kept-edited | 2.3 | `fp_downcast_rounding` legal values `"rtne"`/`"rtz"` added from `core.cast` docstring; "compute in fp32" slogan moved to 17.5 |
| §3.1 Single Program = Single Tile (96–104) | kept | 3.1 | — |
| §3.2 Block Tensors (106–113) | kept-edited | 3.2 | "need not be powers of 2 in principle" replaced by the verified `tl.arange` power-of-two rule; immutability cross-referenced |
| §3.3 Program ID & Num Programs (115–132) | kept | 3.3 | matmul 2D-grid example dropped as redundant |
| §3.4 Grid Launch Syntax (134–151) | kept-edited | 3.4 | "required for autotune" removed; launch options listed from `CUDAOptions` (defaults, `num_warps` power-of-two assert, `num_ctas > 1` requires SM90+); `warmup` mention kept |
| §3.5 Kernel Parameter Rules (153–167) | kept-edited | 3.5 | specialisation claim corrected: only 16-byte divisibility (`tt.divisibility=16`, `backends/compiler.py`) could be verified; "16/8/4-byte" claim dropped |
| §3.6 tl.assume (169–177) | kept | 3.6 | "commonly used in matmul" dropped |
| §4.1 Pointer-Arithmetic Model (182–199) | kept-edited | 4.1 | 2D example now includes its mask; int64 offset note added |
| §4.2 Broadcasting (201–205) | kept | 4.2 | — |
| §4.3 Masking OOB Accesses (207–221) | kept | 4.3 | — |
| §4.4 Block Pointers (223–245) | kept-edited | 4.4 | "compiler can emit TMA or swizzled loads" (unverified performance claim) dropped; `must_use_result` on `advance` verified and stated; mode-exclusivity rule added |
| §4.5 Tensor Descriptors (247–276) | kept-edited | 4.5 | constraints re-derived from `semantic.make_tensor_descriptor` (1–5 dims accepted, docstring says 2–5; last-dim block >= 16 bytes; `nan` padding rejected for ints); descriptor methods and gather/scatter/atomic dtype sets added; host `TensorDescriptor(base, shape, strides, block_shape, padding)` and `from_tensor` verified; "persistent matmul + warp specialization" mention dropped |
| §5.1 tl.load (281–312) | kept | 5.1 | string values re-verified in `semantic.py` lines 908–949 |
| §5.2 tl.store (314–330) | kept | 5.2 | — |
| §5.3 tl.gather / tl.histogram (332–339) | kept-edited | 5.3 | semantics from docstrings (bins of width 1 from 0) |
| §5.4 Atomic Operations (341–368) | kept-edited | 5.4 | dtype rules added from `atom_red_typechecking_impl` / `atomic_cas` (fp16/bf16 only for `add`; int16 rejected; cas width 16/32/64); "from 05-layer-norm" attribution dropped; spin-lock kept as generic API usage |
| §6 Tile Creation (372–391) | kept-edited | 6 | `arange` rule corrected to the enforced one: both bounds constexpr, `end - start` power of two, `<= 1048576`; `-float('inf')` idiom shown via `tl.full` |
| §7 Shape & View Operations (394–413) | kept-edited | 7 | signatures corrected (`reshape(x, *shape, can_reorder)`, `ravel(x, can_reorder)`, `flip(x, dim)`); `join`/`split` semantics from docstrings; standard helpers `sort`, `topk`, `softmax`, `sigmoid`, `swizzle2d`, `cdiv` added (verified) |
| §8.1 Operator Overloads (418–437) | kept | 8.1 | — |
| §8.2 Explicit Arithmetic (439–451) | kept-edited | 8.2 | bf16 up-cast inside `maximum`/`minimum`/`clamp` added (`core.py` 2521–2571) |
| §8.3 Math (453–478) | kept-edited | 8.3 | dtype restriction stated per function from `math.py` `_check_dtype` (`umulhi` is int-only, not listed as fp before); "softmax trick … one FMA saved" (tuning claim) replaced by the plain identity; libdevice pointer added |
| §9 tl.dot (482–518) | kept-edited | 9 | rules rewritten from `semantic.dot`: same-dtype requirement, min sizes (K >= 16 / 32), int8 -> int32, bf16 `out_dtype` rejected, `input_precision` set now includes `bf16x3`/`bf16x6`; "fp16 acc fastest" and "training default" tuning remarks dropped; GEMM loop reduced to an accumulator-chaining illustration (no pointer setup, no grouping) |
| §9.1 tl.dot_scaled (520–533) | kept-edited | 9.1 | shapes/packing from docstring; "sm_100+ native" kept as "hardware without native support emulates" |
| §10 Reductions (536–554) | kept-edited | 10 | **corrected**: `tl.sum` does not promote bf16 (only `max`/`min`/`cumsum`/`cumprod` call `_promote_bfloat16_to_float32`); int<32 promotion verified in `_pick_sum_dtype` |
| §10.1 Custom Reductions (556–574) | kept | 10.1 | — |
| §10.2 Scans (576–583) | kept | 10.2 | — |
| §11.1 Python for / if (588–598) | kept | 11.1 | — |
| §11.2 tl.range (600–615) | kept-edited | 11.2 | `num_stages` default corrected from 3 to `None`; "typical values 2–4" (tuning) dropped; Blackwell-only / matmul-only limits of `warp_specialize` from docstring; grid-stride loop from Pattern F kept here as generic usage |
| §11.3 tl.static_range (617–624) | kept | 11.1 | merged |
| §11.4 While + tl.condition (626–634) | kept | 11.3 | — |
| §12 Debug & Introspection (637–648) | kept | 12 | — |
| §13 Compiler Hints (651–671) | kept-edited | 13 | "persistent matmul" framing removed; generic offset example kept |
| §14 Randomness (674–691) | kept-edited | 15 | signatures corrected (`rand(seed, offset, n_rounds=10)`) |
| §15 @triton.jit (694–723) | kept-edited | 14 | builtin list corrected from `code_generator.builtin_namespace` (`len, list, range, float, int, isinstance, getattr, hasattr, min, max, print`; `copy`/`math` modules not confirmed and dropped); `launch_metadata` kept; specialisation statement aligned with 3.5 |
| §16 @triton.autotune (726–752) | removed | — | autotuning is forbidden in the experiment; one-line notice in the intro only |
| §16.1 triton.Config (754–774) | removed | — | autotune sweep API |
| §16.2 best_config (776–786) | removed | — | autotune workflow |
| §16.3 Disk Cache (788–790) | removed | — | autotune workflow; note the env var named there (`TRITON_AUTOTUNING_CACHE`) does not exist in 3.6.0 (`knobs.py` has `TRITON_CACHE_AUTOTUNING`) |
| §16.4 Verbose Logging (792–799) | removed | — | autotune workflow |
| §17 @triton.heuristics (802–816) | kept-edited | 14.1 | launch-time derived constexpr; not an autotuner; shortened |
| §18 TileBench Integration (819–831) | removed | — | TileBench `run()`/`get_last_config()` contract |
| §18.1 Canonical Template (833–894) | removed | — | TileBench template built on `triton.autotune` |
| §18.2 Why the Dual-Kernel Pattern (896–901) | removed | — | TileBench engine behaviour |
| §18.3 What NOT to do (903–915) | removed | — | TileBench conventions |
| §18.4 Common key= Choices (917–927) | removed | — | autotune `key=` guidance |
| §19 Common Kernel Patterns (931) | removed | — | example-kernel collection |
| Pattern A Element-wise (933–951) | removed | — | solutions to `mul2`, `relu`, `vector_add`, `swiglu`, `dropout`; the generic 1D masked load/store survives as 4.1 |
| Pattern B Row-wise single-pass (953–975) | removed | — | `softmax` solution |
| Pattern C Row-wise two-pass (977–1006) | removed | — | `rmsnorm`/`layernorm`/`l2_norm` solution |
| Pattern D Matmul (1008–1056) | removed | — | full grouped GEMM (`batched_matmul`, `streamk_matmul`, `matmul_*` family); only the generic accumulator chain is illustrated in 9 |
| Pattern E Flash Attention (1058–1133) | removed | — | `flash_attention` solution |
| Pattern F Persistent Kernel (1135–1151) | kept-edited | 11.2 | reduced to the 2-line grid-stride loop (generic API usage); "enables TMA + warp specialization" claim dropped |
| Pattern G Spin-Locked Reduction (1153–1173) | removed | — | layernorm-backward specific; the generic lock idiom is in 5.4 |
| Pattern H make_tensor_descriptor (1175–1194) | kept-edited | 4.5 | folded into the descriptor section (same as the official docstring example) |
| §20 Performance Methodology (1198) | removed | — | tuning methodology |
| §20.1 Occupancy / Resource Tradeoffs (1200–1207) | removed | — | tuning recommendations; the *definitions* of `num_warps`/`num_stages`/`num_ctas`/`maxnreg` survive as launch options in 3.4 |
| §20.2 Saturating HBM Bandwidth (1209–1213) | moved-to-device-context | — | only B200 numbers; **stale** (8 TB/s, ~50 MB L2 vs the repo's measured 6539.4 GB/s and 126.5 MB L2); `driver.active.clear_cache` not carried over |
| §20.3 Compute-Bound Checks (1215–1219) | removed | — | roofline tuning advice |
| §20.4 triton.testing.do_bench (1221–1233) | removed | — | benchmarking API, forbidden setting |
| §20.5 perf_report Pattern (1235–1250) | removed | — | benchmarking API |
| §21 Debugging Checklist (1254–1265) | kept-edited | 17.9 | "Slow vs PyTorch -> autotune" row removed; fp16-integer row moved to 2.4; rows added for verified compiler errors |
| §21.1 Dumping IRs (1267–1272) | kept-edited | 18 | `MLIR_ENABLE_DUMP` confirmed in `libtriton.so` strings; `TRITON_KERNEL_DUMP`/`TRITON_DUMP_DIR` added from `knobs.py` |
| §21.2 Interpreter Mode (1274–1280) | kept | 18 | merged into the env table |
| §22.1 constexpr-ness of Shapes (1286–1288) | kept | 17.1 | — |
| §22.2 tl.advance Returns a New Pointer (1290–1300) | kept | 17.2 | — |
| §22.3 Blocks Are Immutable (1302–1311) | kept | 17.3 | — |
| §22.4 Mask or Pad Non-Pow-2 (1313–1321) | kept | 17.4 | — |
| §22.5 Compute in fp32, Store in Source Dtype (1323–1331) | kept-edited | 17.5 | restated as the dtype rule, not a recommendation |
| §22.6 bfloat16 Promoted in Reductions (1333–1335) | kept-edited | 10 | corrected (not `sum`) |
| §22.7 num_stages on Loops vs Kernels (1337–1342) | kept | 17.7 | "for rmsnorm, softmax" operator mention removed |
| §22.8 triton.cdiv vs tl.cdiv (1344–1354) | kept | 16 | — |
| §22.9 next_power_of_2 Is Host-Only (1356–1363) | kept | 16 | — |
| §22.10 Tensor Pointers vs Block Pointers (1365–1370) | kept | 17.6 | — |
| §22.11 Float8 GEMM (1372–1376) | kept-edited | 17.8 | rewritten from `semantic.dot`; "Blackwell non-transposed V for fp8 fmha / Hopper needs `.T`" could not be verified by introspection and is dropped; title's `ct.mma` reference removed |
| §22.12 Kernel Must Not Mutate Inputs (1378–1380) | removed | — | `restore_value` is an autotune feature |
| §23 TileBench-Specific Workflow (1384) | removed | — | repository workflow |
| §23.1 File Skeleton (1386–1396) | removed | — | TileBench layout (also stale: pre-package path) |
| §23.2 Running & Profiling (1398–1415) | removed | — | `run_bench.py` workflow (stale paths) |
| §23.3 Engine Behavior (1417–1426) | removed | — | engine description (stale) |
| §23.4 config.yaml Example (1428–1452) | removed | — | TileBench config |
| §24.1 tl.load Parameters (1458–1469) | kept | 5.1 | folded into the load table |
| §24.2 tl.store Parameters (1471–1480) | kept | 5.2 | folded |
| §24.3 tl.dot Dtype Matrix (1482–1492) | kept-edited | 9 | replaced by the verified rule list (the source's "fp16 acc -> fp16 or fp32 output" was ambiguous) |
| §24.4 triton.Config Fields (1494–1504) | removed | — | autotune object; the launch options themselves are in 3.4 |
| §24.5 @triton.autotune Args (1506–1516) | removed | — | autotune |
| §24.6 tl.range Attributes (1518–1527) | kept | 11.2 | folded into the signature comment |
| §25 Cheatsheet (1531) | removed | — | minimal kernels = TileBench operators |
| Vector Add (1533–1544) | removed | — | `vector_add` solution |
| Fused Softmax (1546–1557) | removed | — | `softmax` solution |
| RMSNorm (1559–1581) | removed | — | `rmsnorm` solution |
| Matmul with Grouped Scheduling (1583–1585) | removed | — | "strong starting point" tile sizes = tuning claim |
| Flash Attention Forward (1587–1589) | removed | — | starting configs = tuning claim |
| §26 When cuTile and Triton Disagree (1593–1608) | removed | — | TileBench dual-implementation table |
| §27 Environment Variables (1612–1623) | kept-edited | 18 | autotune variables removed; `TRITON_AUTOTUNING_CACHE` does not exist in 3.6.0; `TRITON_KERNEL_DUMP`, `TRITON_DUMP_DIR`, `LLVM_IR_ENABLE_DUMP`, `TRITON_OVERRIDE_ARCH` added (verified) |
| §28 Further Reading (1627–1644) | removed | — | points to `lib/Triton/triton-3.6.0/` (absent) and to TileBench operator implementations |

Counts: the table has 100 rows = the 107 inventory headings minus the 8 pure container headings (§2, §3, §4, §5, §8, §11, §22, §24, whose action is that of their subsections) plus the frontmatter row. kept 27 · kept-edited 33 · moved-to-device-context 1 · removed 39.

## API symbols verified by introspection of the installed `triton==3.6.0`

Method: `python -c "import triton, triton.language as tl"` with `hasattr`, `dir()`, `inspect.signature` (for `@jit`-wrapped standard functions, the signature of `.fn`), docstrings, and `grep` over `site-packages/triton/{language/semantic.py,language/math.py,language/standard.py,language/core.py,compiler/code_generator.py,backends/compiler.py,backends/nvidia/compiler.py,runtime/jit.py,knobs.py}` plus `strings _C/libtriton.so` for C++-side environment variables. No kernel was compiled or launched.

Host (`triton.*`): `jit`, `heuristics`, `cdiv`, `next_power_of_2`, `compile`, `constexpr_function`, `set_allocator`, `JITFunction` (`.warmup`, `.run`, `__getitem__`), `KernelInterface`, `reinterpret`, `TensorWrapper`, `OutOfResources`, `InterpreterError`, `MockTensor`, `CompilationError`, `TritonError`, `AsyncCompileMode`, `FutureKernel`; `triton.runtime.driver` (`.active.get_device_capability`, `.get_current_target`, `.clear_cache`), `triton.runtime.{JITFunction, Config, Autotuner, Heuristics}`; `triton.tools.tensor_descriptor.TensorDescriptor` (`__init__(base, shape, strides, block_shape, padding='zero')`, `from_tensor(tensor, block_shape, padding='zero')`); `triton.profiler` (importable); `triton.testing.{do_bench, do_bench_cudagraph, assert_close, perf_report, Benchmark}` (exist; intentionally not documented).

Dtypes (`tl.*`): `int1, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, bfloat16, float32, float64, float8e4nv, float8e5, float8e4b8, float8e4b15, float8e5b16`; `dtype.primitive_bitwidth`, `is_int_signed/is_int_unsigned/is_int/is_floating/is_fp8/is_fp16/is_bf16/is_fp32/is_fp64/is_bool/is_ptr`; `tl.constexpr`, `tl.const`, `tl.tensor` (`.to`, `.T`, `.logical_and`, `.logical_or`, `__getitem__` with `None`), `tl.pointer_type`, `tl.block_type`, `tl.tensor_descriptor` (`load, store, atomic_add, atomic_and, atomic_or, atomic_xor, atomic_max, atomic_min, gather, scatter, block_shape, dtype`), `tl.PropagateNan` (`NONE`, `ALL`).

In-kernel functions (`tl.*`): `program_id, num_programs, assume, cast, make_block_ptr, advance, make_tensor_descriptor, load_tensor_descriptor, store_tensor_descriptor, load, store, gather, histogram, atomic_cas, atomic_xchg, atomic_add, atomic_max, atomic_min, atomic_and, atomic_or, atomic_xor, arange, zeros, zeros_like, full, reshape, view, ravel, broadcast, broadcast_to, trans, permute, expand_dims, cat, join, split, interleave, flip, add, sub, mul, maximum, minimum, clamp, where, exp, exp2, log, log2, cos, sin, sqrt, rsqrt, abs, floor, ceil, erf, fma, umulhi, fdiv, sqrt_rn, div_rn, dot, dot_scaled, sum, max, min, argmax, argmin, xor_sum, reduce_or, reduce, cumsum, cumprod, associative_scan, range, static_range, condition, static_print, static_assert, device_print, device_assert, debug_barrier, multiple_of, max_contiguous, max_constancy, rand, randn, randint, rand4x, randn4x, randint4x, cdiv, sigmoid, softmax, sort, topk, swizzle2d, inline_asm_elementwise, philox`; `tl.math.{exp, exp2, log, log2, sqrt, rsqrt, sqrt_rn, div_rn, erf, fma, umulhi, fdiv, abs, floor, ceil, cos, sin}`; `triton.language.extra.libdevice`.

Decorator/launch parameters verified: `triton.jit(fn=None, *, version, repr, launch_metadata, do_not_specialize, do_not_specialize_on_alignment, debug, noinline)`; `CUDAOptions` fields `num_warps=4, num_ctas=1, num_stages=3, maxnreg=None, enable_fp_fusion=True, default_dot_input_precision="tf32", allowed_dot_input_precisions=("tf32","tf32x3","ieee","bf16x3","bf16x6")`; `num_warps` power-of-two assert; `num_ctas > 1` requires SM90+.

String options verified in `semantic.py`: load `cache_modifier` `.ca/.cg/.cv`; store `cache_modifier` `.wb/.cg/.cs/.wt`; `eviction_policy` `evict_first/evict_last`; `padding_option` `zero/nan`; atomic `sem` `acquire/release/acq_rel/relaxed`, `scope` `gpu/cta/sys`; `cast` `fp_downcast_rounding` `rtne/rtz`.

Environment variables verified: `TRITON_INTERPRET, TRITON_CACHE_DIR, TRITON_ALWAYS_COMPILE, TRITON_DEBUG, TRITON_F32_DEFAULT, TRITON_KERNEL_DUMP, TRITON_DUMP_DIR, TRITON_OVERRIDE_ARCH, TRITON_PRINT_AUTOTUNING, TRITON_CACHE_AUTOTUNING` (`knobs.py`); `MLIR_ENABLE_DUMP, LLVM_IR_ENABLE_DUMP` (`libtriton.so`).

## Symbols or claims from the source NOT confirmed in 3.6.0 (removed or corrected)

| Source claim | Finding | Handling |
|---|---|---|
| `TRITON_AUTOTUNING_CACHE=1` (§16.3, §27) | no such knob; 3.6.0 reads `TRITON_CACHE_AUTOTUNING` | removed (autotune anyway) |
| `tl.range(..., num_stages=3)` "default 3" (§11.2) | signature default is `None` | corrected |
| "`bfloat16` is promoted to `float32` before max/min/**sum**/scan" (§10, §22.6) | `tl.sum` does not call `_promote_bfloat16_to_float32`; `max`, `min`, `cumsum`, `cumprod`, `maximum`, `minimum`, `clamp` do | corrected |
| pointer specialisation "on 16/8/4-byte alignment" (§3.5, §15) | only a 16-byte divisibility attribute exists in `backends/compiler.py`; value-based specialisation goes through a C extension (`native_specialize_impl`) and was not inspected | reduced to the 16-byte rule |
| in-kernel builtins "`copy`, `math` module" (§15) | `builtin_namespace` = `len, list, range, float, int, isinstance, getattr, hasattr, print, min, max`; module access is allowed generically for `ModuleType` globals but `copy`/`math` specifically were not confirmed | dropped |
| `tl.dot` fp8 "typically requires fp32 accumulator"; "Blackwell supports non-transposed V for fp8 fmha; Hopper requires `.T`" (§22.11) | fp8 x fp8 accepts `out_dtype` fp16 or fp32; the V-transpose statement is not introspectable | first corrected, second dropped |
| `tl.dot` input dtype table lists `uint8`-free int8 only (§9) | `semantic.dot` accepts `float64` operands as well and rejects `uint8` after the first check | table rewritten |
| `make_tensor_descriptor` "2–5 dimensions" (§4.5) | the semantic check accepts `1 <= ndim <= 5`; the docstring says 2–5 | both stated |
| §20.2 "8 TB/s", "~50 MB L2" on B200 | hardware numbers; inventory marks them stale (measured 6539.4 GB/s, 126.5 MB L2) | moved-to-device-context with the stale flag |
| `tl.dot` "tile_m >= 64 for tcgen05" (not in the source; from repo CLAUDE.md) | not a Python-level constraint; not introspectable | not included; noted for the Device Context author |
| `lib/Triton/triton-3.6.0/...` paths (§28, title) | directory absent from the worktree | removed |

Symbols from the source that were probed and **do exist** but are intentionally not documented because they belong to forbidden workflows: `triton.autotune`, `triton.Config` (`kwargs, num_warps, num_stages, num_ctas, maxnreg, pre_hook, ir_override`), `triton.runtime.Autotuner` (`best_config` not probed), `triton.testing.*`, `triton.profiler`, `TRITON_PRINT_AUTOTUNING`, `TRITON_CACHE_AUTOTUNING`.
