# Revision map: cutile-guide -> cutile-reference 1.5.0

Sources:

- Public: `skills/cutile-guide/SKILL.md`, sha256 `7f25fdf978bd9f82c7a8f7da8f9cb88b9e0d6343ec8a47f41d5916c9516b699a`, 1140 lines, 35,299 bytes.
- Local: `/projects/kzhou6/bcui2/research/tilebench/Tilebench/.claude/skills/cutile-guide/SKILL.md`, sha256 `767f467757f89693262626d0989b39fbe3a908d47ae22378ce71f4e8177c71c9`, 1158 lines, 36,262 bytes. Identical to the public copy except that it inserts `### 18.3 Compile-Time Loop Unrolling — ct.static_iter` and renumbers `18.3 Runtime Conditionals` to `18.4`. The public copy lacks the `ct.static_iter` section entirely; the reference takes that section from the local copy (verified against the installed `static_iter` docstring).
- Installed package checked: `cuda-tile` 1.5.0 (`cuda/tile/__init__.py`, `ct.__version__ == "1.5.0"`), plus `cuda-tile-experimental` 0.0.1 (`cuda/tile_experimental/__init__.py`). Verification = `import`, `dir()`, `inspect.signature`, `inspect.getdoc`, and reading `cuda/tile/_stub.py`, `_datatype.py`, `_compiler_options.py`, `_execution.py`, `_ir/control_flow_ops.py`, `_ir/arithmetic_ops.py`, `_ir/typing_support.py`, `tune/_tune.py`. No kernels were launched.

Section numbers below are the local (`.claude`) copy's; the public copy is identical up to 18.2 and shifted by one subsection from 18.3 on.

## Section-by-section actions

| Source section | Action | Target section | Reason |
|---|---|---|---|
| Frontmatter (`name: cutile-guide`, `user-invocable`, `allowed-tools`, `argument-hint`) | REPLACE | frontmatter | v2 protocol frontmatter (`name/dsl/version/kind/derived_from/revised`); Claude Code fields dropped |
| Title / intro ("targeting Blackwell B200 / sm_100") | REVISE | title + scope note | hardware target moved to Device Context; adds "no autotuning" line and verification scope |
| 1 Import Boilerplate | REVISE | 1 | removed `SimpleNamespace`, `numpy`, `RMd` alias, guarded `ct_experimental` import and `_last_autotune_config` (TileBench/autotune contract); kept `ct.Constant[...]` aliases; added execution-space definitions |
| 2.1 DType Constants | REVISE | 2 | added `float8_e8m0fnu`, `float4_e2m1fn` (new in 1.5.0); added arithmetic-vs-numeric classification from docstrings |
| 2.2 DType Usage in Kernels | REVISE | 2 | numpy/torch dtype acceptance verified against `_ir/typing_support._dtype_registry` (44 entries); folded into one bullet |
| 3.1 Array | REVISE | 4.1 | `shape` is NOT compile-time (old text wrong): runtime int32 unless `ArrayAnnotation(static_shape_dims=...)`; added `tiled_view`, `get_raw_memory`; `slice` kept (signature verified) |
| 3.2 Tile | REVISE | 4.2 | `__getitem__` is expand_dims sugar only (verified in `_stub.py`); added `__matmul__`, `__index__`, builtin `max`/`min` mapping |
| 3.3 Scalar | KEEP | 4.3 | verified `Scalar = int | float | ScalarProtocol` |
| 4.1 PaddingMode | KEEP | 5 | all six members verified |
| 4.2 RoundingMode | REVISE | 5 | added `RZI`; per-function restrictions (`exp`/`tanh` FULL/APPROX only) |
| 4.3 MemoryOrder | REVISE | 5 | added `WEAK`; `load`/`store` now accept `memory_order` |
| 4.4 MemoryScope | REVISE | 5 | added `NONE`, `CLUSTER` |
| 5.1 @ct.kernel | REVISE | 3.1, 3.2 | added `num_worker_warps` (4/8, CTK 13.3+), `replace_hints`, `ByTarget` signature, parameter table incl. `ScalarInt64`, `IndexedWithInt64`, `ArrayAnnotation`, `ListAnnotation`; legal ranges verified in `_compiler_options.py` |
| 5.2 @ct.function | KEEP | 3.5 | signature `(func=None, /, *, host=False, tile=True)` verified |
| 5.3 ct.launch | KEEP | 3.3 | positional-only signature verified |
| 5.4 Block/Grid Information | KEEP | 3.4 | `bid`, `num_blocks`, `num_tiles` verified |
| 6.1 ct.load | REVISE | 6.1 | added `memory_order`/`memory_scope`; formal tile-space semantics and entirely-OOB UB from docstring; `allow_tma` default is `None` (TMA allowed), not `True` |
| 6.2 ct.store | REVISE | 6.2 | "indices must be static" reworded to the verified rule (block-uniform tile-space coordinate; scalars/0-d tiles allowed; per-element -> scatter); added `memory_order`/`memory_scope`, rank rule, scalar broadcast |
| 6.3 ct.gather | REVISE | 6.4 | added `mask` (new); mask AND bounds semantics |
| 6.4 ct.scatter | REVISE | 6.4 | added `mask` (new) |
| (none) | ADD | 6.3, 6.5, 6.6 | `TiledView` (load/store/num_tiles/atomic_store_*/traversal_steps), `load_advanced_indexing`/`store_advanced_indexing`/`Slice`, `RawArrayMemory.load_offset/store_offset`: present in 1.5.0, absent from both guides |
| 7 Atomic Operations | KEEP | 7 | signatures and defaults verified; histogram example dropped (operator example) |
| 8 Tile Creation | REVISE | 8 | `arange(size, *, dtype, start=0, step=1)` (keyword-only dtype, new start/step); added `astile` |
| 9 Shape & View Operations | REVISE | 9 | `cat` requires equal shapes; `extract` index is sub-tile grid coordinate (both from docstrings); added `pack_to_bytes`/`unpack_from_bytes` |
| 10 Type Conversion | REVISE | 10 | kept; added promotion rules (binary ops promote, `mma` does not) |
| 11.1-11.3 Arithmetic / Bitwise | KEEP | 11 | all signatures verified; `floordiv` float support noted |
| 12 Comparison | KEEP | 11 | verified |
| 13.1 Exponential & Logarithmic | REVISE | 12 | `log10`, `log1p` REMOVED (not in 1.5.0); `exp` gained `rounding_mode` |
| 13.2 Trigonometric | REVISE | 12 | added `atan2`; `tanh` `rounding_mode` |
| 13.3 Power & Root | KEEP | 12 | verified |
| 13.4 Rounding | REVISE | 12 | added `isnan` |
| 14 Reductions | KEEP | 13 | verified; axis-tuple restriction for argmax/argmin and tie rule added |
| 14.1 Custom Reduction | REVISE | 13 | `reduce(x, /, axis, func, identity, *, keepdims)`; added `ct.scan`; "no loops / no nesting in func" from `TileSyntaxError` sites |
| 15 Scan | KEEP | 13 | verified |
| 16 Conditional & Selection | KEEP | 14 | `where` promotes and broadcasts scalars (verified in `_ir/arithmetic_ops.where`) |
| 17.1 ct.mma | REVISE | 15.1 | added `use_fast_acc`; dtype table re-verified against `_datatype._mma_supported_dtypes`; int8/uint8 mixing rule; tensor-core threshold stated as device/compiler fact and pointed to Device Context (number moved out) |
| 17.2 ct.matmul | KEEP | 15.2 | supported-dtype list from docstring |
| (none) | ADD | 15.3 | `ct.mma_scaled` with table from `_datatype._mma_scaled_supported_dtypes` |
| 18.1 Python for-loop (compile-time unrolled) | REWRITE | 16.1 | WRONG in both guides: `_ir/control_flow_ops.loop_impl` always emits a runtime `ForOp` (`encode_ForOp`) for `for ... in range(...)`; `while` -> `LoopOp`; `break`/`continue`/`return` ops exist |
| 18.2 Python if/else (compile-time) | REVISE | 16.1 | runtime `IfElse` op for runtime conditions; constant conditions folded |
| 18.3 ct.static_iter (local copy only) | KEEP (verified) | 16.2 | docstring confirms: compile-time, <=1000 items, no break/continue/return, must be the `for` iterable; TileBench `top_k_selection` mention removed |
| 18.4 Runtime Conditionals | MERGE | 16.1 | merged into control-flow section |
| 19 Synchronization (Advanced) | REMOVE | - | `cuda_tile.dialects.cuda_tile_ops` is not importable in this environment (`No module named 'cuda_tile'`); `make_token`, `load_ptr_tko`, `store_ptr_tko`, `join_tokens` unverifiable; listed in SKILL.md section 21 as absent |
| 20 TensorView & PartitionView (Advanced) | REMOVE | - | same module; superseded by `Array.tiled_view`/`TiledView` (section 6.3) |
| 21.1 printf | REVISE | 17 | added `ct.print` (f-string form), `compiler_timeout`, exception list |
| 21.2 assert_ | KEEP | 17 | verified |
| 22 Compiler Hints (Advanced) | REWRITE | 18 | `assume_div_by` RENAMED to `assume_divisible_by(x, divisor)` (no `every`/`along`); `assume_same_elements`, `assume_bounded`, `optimization_barrier` REMOVED (absent) |
| 23 Autotune Pattern (23.1 autotune_launch, 23.2 clear_autotune_cache, 23.3 template) | REMOVE | - | autotuning content; see "deprecated" below for the API status |
| 24 run() and get_last_config() Contract | REMOVE | - | TileBench repository contract |
| 25 Common Kernel Patterns A-F | REMOVE | 20 (replaced) | A/B/C/D/E are full solutions to TileBench operators (mul2, relu, vector_add, dropout, swiglu, softmax, argmax, rmsnorm, layernorm, l2_norm, flash_attention, destindex); F is the TileBench `NotImplementedError` contract. Replaced by three generic API illustrations that are not among the 45 operators (`scale_shift`, `row_abs_max`, `copy_cast`) |
| 26 Critical Constraints & Gotchas | REVISE | 19 | kept pow2, store-vs-scatter, immutability, fp32 accumulation, grid tuple, kernel-vs-host, fp16 integer range; "Autotune has no persistent cache" removed; "flush_to_zero for exp2" folded into 12/19; expanded with verified rules (runtime `Array.shape`, entirely-OOB UB, `cat`/`extract` rules, negative indices, int64 annotations) |
| 27 Quick Reference Tables | MERGE | 6.1, 6.2, 15.1 | folded into the signature blocks; `allow_tma` default corrected to `None` |
| B200/sm_100 mentions throughout (`ByTarget(sm_100=...)` example kept as syntax only) | MOVE OUT | Device Context | hardware numbers and tensor-core tile thresholds belong to the Device Context document |

## Symbols verified present in cuda-tile 1.5.0

Public names on `cuda.tile` (146):

`Array, ArrayAnnotation, ByTarget, Constant, ConstantAnnotation, DType, IndexedWithInt64, ListAnnotation, MemoryOrder, MemoryScope, PaddingMode, RoundingMode, Scalar, ScalarInt64, Slice, Tile, TileCompilerExecutionError, TileCompilerTimeoutError, TileError, TileInternalError, TileRecursionError, TileStaticAssertionError, TileStaticEvalError, TileSyntaxError, TileTypeError, TileUnsupportedFeatureError, TileValueError, TiledView, abs, add, arange, argmax, argmin, assert_, assume_divisible_by, astile, astype, atan2, atomic_add, atomic_and, atomic_cas, atomic_max, atomic_min, atomic_or, atomic_xchg, atomic_xor, bfloat16, bid, bitcast, bitwise_and, bitwise_lshift, bitwise_not, bitwise_or, bitwise_rshift, bitwise_xor, bool_, broadcast_to, cat, cdiv, ceil, compilation, compiler_timeout, cos, cosh, cumprod, cumsum, equal, exp, exp2, expand_dims, extract, float16, float32, float4_e2m1fn, float64, float8_e4m3fn, float8_e5m2, float8_e8m0fnu, floor, floordiv, full, function, gather, greater, greater_equal, int16, int32, int64, int8, isnan, kernel, launch, less, less_equal, load, load_advanced_indexing, log, log2, matmul, max, maximum, min, minimum, mma, mma_scaled, mod, mul, negative, not_equal, num_blocks, num_tiles, ones, pack_to_bytes, permute, pow, print, printf, prod, reduce, reshape, rsqrt, scan, scatter, sin, sinh, sqrt, static_assert, static_eval, static_iter, store, store_advanced_indexing, sub, sum, tan, tanh, tfloat32, transpose, truediv, tune, uint16, uint32, uint64, uint8, unpack_from_bytes, where, zeros`

Methods/attributes verified: `Array.{dtype, ndim, shape, strides, slice, tiled_view, get_raw_memory}`; `Tile.{dtype, shape, ndim, astype, reshape, permute, transpose, extract, item, __getitem__, __matmul__, __index__, arithmetic/bitwise/comparison dunders}`; `TiledView.{dtype, tile_shape, traversal_steps, num_tiles, load, store, atomic_store_add, atomic_store_and, atomic_store_max, atomic_store_min, atomic_store_or, atomic_store_xor}`; `RawArrayMemory.{load_offset, store_offset}` (`cuda.tile._stub`); kernel object method `replace_hints(**hints)`; `ct.kernel` options `num_ctas, occupancy, opt_level, num_worker_warps`; `ct.tune.{exhaustive_search, TuningResult, Measurement}` (present, intentionally undocumented in the reference); `cuda.tile_experimental.{autotune_launch, clear_autotune_cache}` (present in the separate 0.0.1 package, intentionally undocumented).

Of these, the reference documents everything except: `compilation` (internal module), `jax` submodule, `tune`, `TileCompilerExecutionError`-family internals beyond their names, and `ConstantAnnotation`/`ArrayAnnotation` internals beyond their constructor arguments.

## Symbols used by the guides that are NOT found in 1.5.0, or are deprecated

| Guide symbol | Status in 1.5.0 | Replacement / note |
|---|---|---|
| `ct.log10`, `ct.log1p` | absent | compose from `ct.log`/`ct.log2` |
| `ct.assume_div_by(x, divisor=, every=, along=)` | absent (renamed) | `ct.assume_divisible_by(x, divisor)` (scalar only, no `every`/`along`) |
| `ct.assume_same_elements` | absent | none |
| `ct.assume_bounded` | absent | none |
| `ct.optimization_barrier` | absent | none |
| `cuda_tile.dialects.cuda_tile_ops.{make_token, join_tokens, load_ptr_tko, store_ptr_tko, load_view_tko, store_view_tko, make_tensor_view, make_partition_view, get_tensor_shape}` | module `cuda_tile` not importable | `Array.tiled_view` / `TiledView` for tiled views; load/store `memory_order`/`memory_scope` for ordering |
| `ct_experimental.autotune_launch(...)` -> `TunedResult` (`.tuned_config`, `.grid`, `.cache_hit`, `.tuning_record`) | present only in `cuda.tile_experimental` 0.0.1 (git-installed, not on PyPI); the installed source carries no `DeprecationWarning`, but it is the older tune-and-launch API superseded by `ct.tune.exhaustive_search` (returns `TuningResult` with `.best.config`, `.best.mean_us`, `.successes`, `.failures`, `.summary()`, and only tunes; the caller launches) | both excluded from the reference (autotuning not permitted) |
| `ct_experimental.clear_autotune_cache` | same package | excluded |
| top-level `import ct_experimental` | absent | `import cuda.tile_experimental` (irrelevant here) |
| `Array.shape` "compile-time per axis" | incorrect claim | runtime value; `ArrayAnnotation(static_shape_dims=...)` for constants |
| "Python for-loop (compile-time unrolled)" (18.1) | incorrect claim | runtime `ForOp`; `ct.static_iter` is the unrolled form |
| `ct.load(..., allow_tma=True)` default | default is `None` (TMA allowed) | documented as `None` |
| `RMd` alias (`from cuda.tile import RoundingMode as RMd`) | valid Python but not an API name | reference uses `ct.RoundingMode` |

## Items deliberately moved to the Device Context document

- Target architecture naming (B200, sm_100) and all SM/L2/bandwidth numbers.
- Minimum `mma` tile dimensions that select tensor-core (tcgen05 / HMMA) lowering; the reference states only that this is a compiler/device decision.
- Availability of `mma_scaled` and `num_worker_warps` by architecture/CTK version is stated as "architecture dependent" without numbers.

## Counts

- Source sections (local copy): 27 top-level, 59 headings including subsections. Kept or revised: 44; removed: 13 (19, 20, 23, 23.1, 23.2, 23.3, 24, 25 A-F, 26 "Autotune has no persistent cache"); added: 5 (TiledView, advanced indexing, RawArrayMemory, mma_scaled, scan).
- Reference: 21 numbered sections, 3 generic examples, 0 TileBench operator solutions, 0 hardware numbers, 0 autotune APIs.
