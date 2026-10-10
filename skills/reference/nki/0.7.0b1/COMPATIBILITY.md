# NKI 0.7.0b1 compatibility matrix (trn2, NeuronCore-v3, LNC=2)

Companion to `SKILL.md` (not injected). Every public symbol of `nki`, `nki.language`, `nki.isa`, `nki.compiler` and `nki.collectives` in the installed package (`evidence/introspection.json`), plus names that older public material uses.

**Status legend.**

| Code | Meaning |
|---|---|
| S | supported; behaviour verified on the trn2 device (evidence column) |
| R | supported with a restriction stated in the note; restriction verified on device |
| D | deprecated in 0.7.0b1 docstrings or release notes but still works (verified where an evidence case is given) |
| A | absent in 0.7.0b1 (introspection) or rejected as removed |
| G4 | present but NeuronCore-v4 (trn3) only; rejected on trn2 |
| P | present in the installed package; signature/docstring only, **not exercised on the device** |

**Evidence.** `test::case` refers to `evidence/smoke_results.json` (script `evidence/smoke_tests/<test>.py`). `pkg` refers to the installed source/docstring (`evidence/introspection.json` -> `modules.<module>.symbols.<name>`; source path relative to site-packages in the `source` field). Device results: nki `0.7.0b1+34883233359.g2aa6a82a`, neuronx-cc `2.0.404830.0a0+55f4bf89`, torch-neuronx `2.14.3.0.0+unknown.dev`, trn2.3xlarge, `NEURON_LOGICAL_NC_CONFIG=2`.

## Summary (counts per status)

| Module | S | R | D | A | G4 | P | total |
|---|---|---|---|---|---|---|---|
| `nki.language` | 42 | 5 | 4 | 0 | 7 | 87 | 145 |
| `nki.isa` | 15 | 10 | 0 | 0 | 5 | 41 | 71 |
| `nki` | 2 | 0 | 0 | 0 | 0 | 3 | 5 |
| `nki.compiler` | 0 | 0 | 0 | 0 | 0 | 8 | 8 |
| `nki.collectives` | 0 | 0 | 0 | 0 | 0 | 17 | 17 |
| `legacy (absent)` | 0 | 0 | 0 | 16 | 0 | 0 | 16 |
| **all** | 59 | 15 | 4 | 16 | 12 | 156 | 262 |

Module rows count the public names of `dir(module)` (sub-modules and re-exported helpers such as `Union`/`warnings` excluded). The last row counts names from older public material that are absent or rejected. Section 10 of `SKILL.md` is the condensed version of this matrix.

## Behaviour differences: installed 0.7.0b1 vs public docs, agentic skills and samples

The "Public source" column cites aws-neuron/neuron-agentic-development@`ee25e45` (paths under `skills/`), the AWS Neuron docs (Neuron 2.32.0 / NKI 0.6.0, accessed 2026-10-10) or aws-neuron/nki-samples@`62c300da`. The installed package and the device test decide.

| # | Topic | Public source says | Installed 0.7.0b1 on trn2 | Evidence |
|---|---|---|---|---|
| 1 | Default LNC | `nki.jit` docstring: "The default is LNC=1"; mismatch with `NEURON_LOGICAL_NC_CONFIG` causes a runtime error. `neuron-nki-docs/references/programming/lnc.md`: no brackets means 1 core | Unset LNC resolves to `NEURON_LOGICAL_NC_CONFIG` (2 here): bare call has `num_programs()==2`, `program_ndim()==1`. `kernel[1]` runs (1 program, `program_ndim()==0`) without a runtime error | pkg `nki/framework/_lnc.py`; `b_lnc2_program_id::probe_bare_call`, `::probe_lnc1_bracket`; `a_elementwise_add_launch::lnc1_bracket` |
| 2 | Default frontend | `neuron-nki-writing/references/nki-language-constraint.md`, `nki-060-dynamic-loop-migration-guide.md`: tracer default in 0.7.0, parser removed in 0.8.0 | Default is still `custom` (= parser); `standard` (= tracer) is opt-in via `NKI_COMPILER_FRONTEND` | pkg `nki/_config.py` (`compiler_frontend`), `nki/compiler/frontend.py` (`resolve_frontend_cls`) |
| 3 | `nl.dynamic_range` | 0.6.0 release notes: "NKI emits a deprecation warning" | Compiles and runs correctly under the default frontend; no Python warning and no compiler warning captured | `f_dynamic_range_and_ranges::dynamic_range_int_bound` |
| 4 | `nl.load` / `nl.store` | `nki-language-constraint.md`: Beta-1 API, "NEVER use" | Present; experimental wrappers around `nisa.dma_copy`; work on device; `mask=` rejected | pkg `nki/language/_core.py`; `j_load_store::*`; `d_negative_limits::removed_mask_kwarg` |
| 5 | `affine_range` vs `sequential_range` | `nki-language-constraint.md`: use `affine_range`, NOT `sequential_range`, for PSUM accumulation loops | All `nl.*_range` are documented as fully unrolled aliases of `range()` and marked deprecated; all gave correct loop-carried results, no warnings | pkg `nki/language/_ranges.py`; `f_dynamic_range_and_ranges::loop_carried_*` |
| 6 | `nisa.rsqrt` | `nki-language-constraint.md`: replace `activation(op=nl.rsqrt)` by `nisa.rsqrt` | `nisa.rsqrt` does not exist; `nl.rsqrt` is a valid activation op | introspection `modules.nki.isa.dir_public` |
| 7 | `tensor_reduce(negate=...)` | `nki-language-constraint.md`: remove `negate=` | `negate=` exists and works | `g_compute_ops::tensor_reduce_free_axes` |
| 8 | `nl.max` / `nl.min` | `nki-language-constraint.md`: use `nl.maximum` / `nl.minimum` instead | Both exist with different meanings: `nl.max(x, axis)` is a reduction, `nl.maximum(a, b)` is element-wise | pkg `nki/language/_math_funcs.py`, `_operators.py` |
| 9 | `nl.tile_size.psum_fmax_bytes` | `nki-language-constraint.md` (0.4.0 table) | Absent; use `psum_bank_fmax_bytes` (2048). `psum_fmax` and `total_available_sbuf_size` deprecated | pkg `nki/language/_tile_size.py` |
| 10 | `memset` value dtype | `nki-language-constraint.md`: value dtype must match dst (`value=2` not `2.0` for int) | `memset(dst=int32_tile, value=2.0)` compiled and stored 2 | `l_python_subset::memset_float_value_into_int32_tile` |
| 11 | `@nki.jit` on sub-functions | `nki-language-constraint.md`: remove the decorator from helpers | A nested `@nki.jit` call is inlined into the caller (config-conflict check); plain helpers also work | pkg `nki/framework/kernel.py` (`__call__`); `l_python_subset::nested_jit_subkernel_call` |
| 12 | Identity operators | `nki-language-constraint.md`: use `==`/`!=` instead of `is`/`is not` | `is None` / `is not None` in user code work; library code that uses `is not` on other objects prints a warning | `l_python_subset::is_none_user_code`, `::is_not_none_user_code` |
| 13 | Partial tiles | Beta-1 docs/samples: `mask=` | slices starting at partition 0 | `p_shapes_partial_tiles::remainder_rows_200x300` |
| 14 | PSUM accumulation | `neuron-nki-docs/references/architecture/trainium_inferentia2_arch.md`: implicit `+=` into `nl.zeros` PSUM | `nc_matmul(dst=..., accumulate=...)`; `accumulate=None` auto-infers overwrite-then-accumulate | pkg `nki/isa/_matmul.py`; `c_nc_matmul_accumulate::bf16_accumulate_none_auto` |
| 15 | `nl.par_dim`, `nl.arange`, `nl.mgrid`, `nl.spmd_dim`, `nl.nc` | used in `neuron-nki-docs/references/architecture/trainium2_arch.md`, `programming/indexing-overview.md`, tutorials (`fused_mamba.md`, `average_pool2d.md`, `spmd_multiple_nc_tensor_addition.md`) | Absent: `failed to resolve name 'nki.language.arange'` / `'nki.language.par_dim'` | `d_negative_limits::removed_nl_arange`, `::removed_nl_par_dim`; introspection `legacy_names_present` |
| 16 | `nisa.constants.engine.tensor` | `trainium2_arch.md` | `nki.isa.constants` exports only `dge_mode`, `oob_mode`, `matmul_perf_mode`; use `nisa.engine.tensor` | pkg `nki/isa/constants.py` |
| 17 | Framework driver | `programming/framework_custom_op.md`: `torch_xla` / `xm.xla_device()`; nki-samples tutorials: `torch_neuronx.trace(...)` | Native PyTorch: `tensor.to("neuron")` and a direct kernel call; torch-xla not installed | introspection `versions.torch_xla_importable=false`; `a_elementwise_add_launch::bare_call` |
| 18 | `dma_transpose` 2-byte dtype | docstring (DMA direct transpose constraints) | The 2-byte and `src.shape[0]==16` rules apply to `dge_mode=hwdge`; with the default `dge_mode` an fp32 `[128, 64]` transpose worked | `m_data_movement::dma_transpose_fp32_default`, `::dma_transpose_fp32_hwdge` |
| 19 | `nl.matmul` | docstring: "x @ y matrix multiplication" (no dtype caveat) | Allocates its PSUM result with the input dtype; bf16 inputs fail on trn2 (`nc_matmul dst dtype must be float32 on gen3`) | `c_nc_matmul_accumulate::nl_matmul_bf16_inputs` |
| 20 | Kernel without return value | `nki-language-constraint.md` mentions `nki.typing` mutable outputs (caller allocates) | Under torch-neuronx native dispatch a kernel that returns nothing fails (`Cannot call .result on operation torch.operator which has 0 results`) | `m_data_movement::kernel_without_return` |
| 21 | Python subset | (not documented for the parser in the fetched docs) | Rejected: comprehensions, generator expressions, lambda, try, keyword-only params, direct nested-def calls, closures over enclosing locals | `l_python_subset::*`, `k_cache_specialization::closure_constant_factory` |
| 22 | Fori-loop index arithmetic | migration guide: "Do not ... perform Python arithmetic on it" | Confirmed: `i * P` fails (`'mul' expected (int, int) ... got (object, int)`); use `step=P` + `nl.ds(i, P)`, `view.reshape(...)[i]` or `.ap(scalar_offset=i)` | `e_fori_loop_dynamic_offset::*` |
| 23 | Compute-engine partition start | (not stated in the fetched docs) | Operand partition slices must start at 0/32/64/96 (`Expected 'dst' partition start in {0, 32, 64, 96} ...`); DMA can start anywhere | `m_data_movement::partition_slice_*`, `::dma_copy_into_arbitrary_partition_offset` |
| 24 | `nl.right_shift` on int32 | (not stated) | Logical shift (matches an unsigned shift, not torch's arithmetic `>>`) | `o_int_dtypes_creation_capacity::int32_right_shift_const` |
| 25 | `jit` keyword `lnc=` | some older material passes the LNC to the decorator | `TypeError` ("Set LNC at the callsite with bracket syntax") | pkg `nki/__init__.py` |


## Per-symbol matrix

### `nki.language`

| Name | Status | Signature (installed) | Note | Evidence |
|---|---|---|---|---|
| `DType` | P | `` | docstring: str(object='') -> str str(bytes_or_buffer[, encoding[, errors]]) -> str | pkg |
| `DynamicSlice` | P | `(*args: Any, **kwargs: Any) -> Any` | docstring: Base class for NKI kernel dataclasses and configuration objects | pkg |
| `MemoryRegion` | P | `(*values)` | docstring: Memory region constants for NKI tensors | pkg |
| `NKIObject` | P | `(*args: Any, **kwargs: Any) -> Any` | docstring: Base class for NKI kernel dataclasses and configuration objects | pkg |
| `NkiBlockGrid` | P | `(*args: Any, **kwargs: Any) -> Any` | new in 0.7.0b1 (tiled-grid helpers); not covered by SKILL.md | pkg |
| `NkiGrid` | P | `(*args: Any, **kwargs: Any) -> Any` | new in 0.7.0b1 (tiled-grid helpers); not covered by SKILL.md | pkg |
| `NkiTensor` | S | `(*args: Any, **kwargs: Any) -> Any` | tensor/view class: reshape, permute, slice, select, view(dtype), ap(...), indirect (v4) | `m_data_movement::hbm_reshape_and_strided_views`, `m_data_movement::ap_pattern_offset` |
| `NkiTileGrid` | P | `(*args: Any, **kwargs: Any) -> Any` | new in 0.7.0b1 (tiled-grid helpers); not covered by SKILL.md | pkg |
| `abs` | P | `(x, y=None, dtype=None)` | docstring: Absolute value of the input, element-wise | pkg |
| `abs_max` | G4 | `(x, y=None, dtype=None)` | docstring: NeuronCore-v4+ | pkg |
| `abs_min` | G4 | `(x, y=None, dtype=None)` | docstring: NeuronCore-v4+ | pkg |
| `add` | S | `(x, y=None, dtype=None)` | allocates result in the data tile's buffer | `g_compute_ops::nl_convenience_ops` |
| `affine_range` | D | `(start, stop=None, step=1)` | docstring: unrolled alias of range(), deprecated; correct for loop-carried updates, no warning | `f_dynamic_range_and_ranges::loop_carried_affine_range` |
| `all` | P | `(x, axis, dtype=None)` | experimental convenience composed of nisa instructions; allocates SBUF temporaries | pkg |
| `alloc_blocks` | P | `(*, tile_shape: tuple[int, ...], block_shape: tuple[int, ...], element_shape: tuple[int, ...], buffer: nki.language.buffers.MemoryRegion, dtype: str,  ...` | new in 0.7.0b1 (tiled-grid helpers); not covered by SKILL.md | pkg |
| `alloc_tiles` | P | `(*, tile_shape: tuple[int, ...], element_shape: tuple[int, ...], buffer: nki.language.buffers.MemoryRegion, dtype: str, address: tuple \| None = None, ...` | new in 0.7.0b1 (tiled-grid helpers); not covered by SKILL.md | pkg |
| `arctan` | P | `(x, y=None, dtype=None)` | docstring: Inverse tangent of the input, element-wise | pkg |
| `average` | P | `(x, y=None, dtype=None)` |  | pkg |
| `bfloat16` | S | `` |  | `c_nc_matmul_accumulate::bf16_k256_accumulate_explicit_fp32_out` |
| `bitwise_and` | S | `(x, y=None, dtype=None)` | bit-exact on int32 | `o_int_dtypes_creation_capacity::int32_bitwise_and_const` |
| `bitwise_or` | P | `(x, y=None, dtype=None)` | docstring: Compute the bitwise OR of two tiles element-wise | pkg |
| `bitwise_xor` | P | `(x, y=None, dtype=None)` | docstring: Compute the bitwise XOR of two tiles element-wise | pkg |
| `blocks` | P | `(source: nki.language.tensor.NkiTensor \| nki.language._grid.grid.NkiTileGrid \| nki.language._grid.grid.NkiBlockGrid, *, tile_shape: tuple[int, ...]  ...` | new in 0.7.0b1 (tiled-grid helpers); not covered by SKILL.md | pkg |
| `bool_` | P | `` | docstring: str(object='') -> str str(bytes_or_buffer[, encoding[, errors]]) -> str | pkg |
| `broadcast_to` | S | `(x, shape, dtype=None)` | partition broadcast of [1,F] | `q_partition_broadcast::nl_broadcast_to` |
| `bypass` | P | `(x, y=None, dtype=None)` |  | pkg |
| `ceil` | P | `(x, y=None, dtype=None)` | docstring: Ceiling of the input, element-wise | pkg |
| `copy` | S | `(x, y=None, dtype=None)` | op name for activation / tensor copy helper | `i_dtype_cast::activation_copy_to_bf16` |
| `cos` | P | `(x, y=None, dtype=None)` | docstring: Cosine of the input, element-wise | pkg |
| `device_print` | P | `(print_prefix, tensor: 'NkiTensor')` | run-time print via the NRT debug stream (NEURON_RT_DEBUG_OUTPUT_DIR); Python print() runs at trace time | pkg |
| `divide` | P | `(x, y=None, dtype=None)` | docstring: Divide the inputs, element-wise | pkg |
| `dropout` | P | `(x, rate, dtype=None)` | experimental convenience composed of nisa instructions; allocates SBUF temporaries | pkg |
| `ds` | S | `(start: 'Union[int, NkiTensor, VirtualRegister, NkiScalar]', size: 'int')` | `nl.ds(start, size)`; start may be a runtime value (fori index) | `e_fori_loop_dynamic_offset::fori_loop_step_nl_ds` |
| `dynamic_range` | D | `(start, stop=None, step=1)` | works under the default (parser) frontend, no warning observed; deprecated by 0.6.0 notes, removed under the tracer | `f_dynamic_range_and_ranges::dynamic_range_int_bound` |
| `empty_like` | S | `(x: nki.language.tensor.NkiTensor \| nki.language._grid.grid.NkiTileGrid \| nki.language._grid.grid.NkiBlockGrid, dtype: str \| None = None, buffer: n ...` | uninitialised; same shape/dtype/buffer | `o_int_dtypes_creation_capacity::creation_helpers` |
| `equal` | P | `(x, y=None, dtype=None)` | docstring: Return (x == y) element-wise | pkg |
| `erf` | P | `(x, y=None, dtype=None)` | docstring: Error function, element-wise | pkg |
| `erf_dx` | P | `(x, y=None, dtype=None)` | docstring: Derivative of error function, element-wise | pkg |
| `exp` | S | `(x, y=None, dtype=None)` | unary -> activation | `g_compute_ops::nl_convenience_ops` |
| `expand_dims` | P | `(x, axis)` | docstring: Expand the shape of a tile | pkg |
| `float16` | S | `` |  | `a_elementwise_add_launch::bare_call` |
| `float32` | S | `` |  | `i_dtype_cast::tensor_copy_bf16_to_fp32` |
| `float4_e2m1fn_x4` | G4 | `` | MX packed type for v4 instructions | pkg |
| `float8_e4m3` | S | `` | legacy fp8; TensorE input | `d_negative_limits::nc_matmul_float8_e4m3_legacy_tiles` |
| `float8_e4m3fn` | G4 | `` | dtype exists; torch arg rejected on trn2; no TensorE use on v3 | `d_negative_limits::torch_float8_e4m3fn_input`, `d_negative_limits::nc_matmul_float8_e4m3fn_tiles` |
| `float8_e4m3fn_x4` | G4 | `` | MX packed type for v4 instructions | pkg |
| `float8_e5m2` | S | `` | TensorE input; torch e5m2 passes through | `d_negative_limits::nc_matmul_float8_e5m2_tiles` |
| `float8_e5m2_x4` | G4 | `` | MX packed type for v4 instructions | pkg |
| `float8_e8m0fnu` | G4 | `` | MX scale type | pkg |
| `floor` | P | `(x, y=None, dtype=None)` | docstring: Floor of the input, element-wise | pkg |
| `fmod` | P | `(x, y=None, dtype=None)` | docstring: Floating-point remainder of `x / y`, element-wise | pkg |
| `fori_loop` | R | `(lower, upper, body_fun, step=1)` | rolled loop; no Python arithmetic on the index under the default frontend | `e_fori_loop_dynamic_offset::fori_loop_step_nl_ds`, `e_fori_loop_dynamic_offset::fori_loop_python_arith_on_i`, `e_fori_loop_dynamic_offset::fori_loop_register_upper_bound` |
| `full` | S | `(shape, fill_value, dtype, buffer=nl.sbuf, name='')` | fills via memset; shared_hbm rejected | `o_int_dtypes_creation_capacity::creation_helpers` |
| `gather_flattened` | P | `(data: 'NkiTensor', indices: 'NkiTensor', axis=0, dtype=None)` | docstring: Gather elements from data tensor using indices after flattening | pkg |
| `gelu` | P | `(x, y=None, dtype=None)` | docstring: GELU activation, element-wise | pkg |
| `gelu_apprx_sigmoid` | P | `(x, y=None, dtype=None)` | docstring: GELU approximation using sigmoid, element-wise | pkg |
| `gelu_apprx_sigmoid_dx` | P | `(x, y=None, dtype=None)` | docstring: Derivative of sigmoid-approximated GELU, element-wise | pkg |
| `gelu_apprx_tanh` | P | `(x, y=None, dtype=None)` | docstring: GELU approximation using tanh, element-wise | pkg |
| `gelu_dx` | P | `(x, y=None, dtype=None)` | docstring: Derivative of GELU activation, element-wise | pkg |
| `greater` | S | `(x, y=None, dtype=None)` | comparison -> 1/0 | `o_int_dtypes_creation_capacity::compare_to_uint8_and_fp32` |
| `greater_equal` | P | `(x, y=None, dtype=None)` | docstring: Return (x >= y) element-wise | pkg |
| `hbm` | S | `` | alias of private_hbm | introspection `extra.nl.hbm_is_private_hbm` |
| `int16` | P | `` | docstring: str(object='') -> str str(bytes_or_buffer[, encoding[, errors]]) -> str | pkg |
| `int32` | S | `` | Vector/GpSimd; exact on GpSimd | `o_int_dtypes_creation_capacity::int32_add_default_engine` |
| `int8` | R | `` | data movement only; no TensorE | `o_int_dtypes_creation_capacity::torch_int8_roundtrip`, `d_negative_limits::nc_matmul_int8` |
| `invert` | P | `(x, y=None, dtype=None)` | docstring: Compute the bitwise NOT element-wise | pkg |
| `is_hbm` | P | `(buffer)` | docstring: Check if buffer is any HBM type | pkg |
| `is_on_chip` | P | `(buffer)` | docstring: Check if buffer is on-chip (SBUF or PSUM) | pkg |
| `is_psum` | P | `(buffer)` | docstring: Check if buffer is PSUM | pkg |
| `is_sbuf` | S | `(buffer)` |  | `g_compute_ops::nl_convenience_ops` |
| `left_shift` | P | `(x, y=None, dtype=None)` | docstring: Left shift the bits of x by y positions element-wise | pkg |
| `less` | P | `(x, y=None, dtype=None)` | docstring: Return (x < y) element-wise | pkg |
| `less_equal` | P | `(x, y=None, dtype=None)` | docstring: Return (x <= y) element-wise | pkg |
| `load` | S | `(src: 'NkiTensor', dtype=None)` | experimental thin wrapper: SBUF tile of src.shape + dma_copy; `dtype=` converts; no `mask=` | `j_load_store::nl_load_store_tiled`, `j_load_store::nl_load_dtype_conversion`, `d_negative_limits::removed_mask_kwarg` |
| `load_transpose2d` | S | `(src: 'NkiTensor', dtype=None)` | wrapper of dma_transpose; bf16 and fp32 worked (default dge_mode) | `j_load_store::nl_load_transpose2d_bf16`, `j_load_store::nl_load_transpose2d_fp32` |
| `log` | P | `(x, y=None, dtype=None)` | docstring: Natural logarithm of the input, element-wise | pkg |
| `logical_and` | P | `(x, y=None, dtype=None)` | docstring: Compute the logical AND of two tiles element-wise | pkg |
| `logical_not` | P | `(x, y=None, dtype=None)` | docstring: Compute the logical NOT element-wise | pkg |
| `logical_or` | P | `(x, y=None, dtype=None)` | docstring: Compute the logical OR of two tiles element-wise | pkg |
| `logical_xor` | P | `(x, y=None, dtype=None)` | docstring: Compute the logical XOR of two tiles element-wise | pkg |
| `matmul` | R | `(x, y, transpose_x=False)` | allocates PSUM result with x.dtype -> only fp32 inputs work on trn2 | `c_nc_matmul_accumulate::nl_matmul_fp32_inputs`, `c_nc_matmul_accumulate::nl_matmul_bf16_inputs` |
| `max` | P | `(x, axis, dtype=None, keepdims=False)` | reduction `(x, axis, dtype=None, keepdims=False)`; NOT the element-wise maximum/minimum | pkg |
| `maximum` | S | `(x, y=None, dtype=None)` | element-wise (vs nl.max = reduction) | `g_compute_ops::nl_convenience_ops` |
| `mean` | P | `(x, axis, dtype=None, keepdims=False)` | experimental convenience composed of nisa instructions; allocates SBUF temporaries | pkg |
| `min` | P | `(x, axis, dtype=None, keepdims=False)` | reduction `(x, axis, dtype=None, keepdims=False)`; NOT the element-wise maximum/minimum | pkg |
| `minimum` | P | `(x, y=None, dtype=None)` | element-wise minimum (op name / convenience) | pkg |
| `mish` | P | `(x, y=None, dtype=None)` | docstring: Mish activation, element-wise | pkg |
| `mod` | P | `(x, y=None, dtype=None)` | docstring: Remainder of `x / y`, element-wise | pkg |
| `multiply` | S | `(x, y=None, dtype=None)` | scalar / [P,1] operand -> tensor_scalar | `g_compute_ops::nl_convenience_ops` |
| `ndarray` | S | `(shape, dtype, buffer=nl.sbuf, name='', address=None)` | `(shape, dtype, buffer=nl.sbuf, name='', address=None)`; SBUF/PSUM need >=2 dims | `a_elementwise_add_launch::bare_call` |
| `negative` | P | `(x, y=None, dtype=None)` | docstring: Numerical negative of the input, element-wise | pkg |
| `no_reorder` | S | `()` | context manager | `f_dynamic_range_and_ranges::no_reorder_block` |
| `not_equal` | P | `(x, y=None, dtype=None)` | docstring: Return (x != y) element-wise | pkg |
| `num_programs` | S | `(axes=0)` | 2 for bare call / [2]; 1 for [1] | `b_lnc2_program_id::probe_bare_call`, `b_lnc2_program_id::probe_lnc1_bracket` |
| `ones` | S | `(shape, dtype, buffer=nl.sbuf, name='')` | fills via memset | `o_int_dtypes_creation_capacity::creation_helpers` |
| `power` | P | `(x, y=None, dtype=None)` | docstring: Elements of x raised to powers of y, element-wise | pkg |
| `prelu` | P | `(x, y=None, dtype=None)` |  | pkg |
| `private_hbm` | S | `` | per-program HBM (docstring); not device-tested | pkg |
| `prod` | P | `(x, axis, dtype=None, keepdims=False)` | experimental convenience composed of nisa instructions; allocates SBUF temporaries | pkg |
| `profiler_region_close` | P | `()` | new in 0.7.0b1; profiler annotations | pkg |
| `profiler_region_open` | P | `(label, color='')` | new in 0.7.0b1; profiler annotations | pkg |
| `program_id` | S | `(axis=0)` | trace-time int; 0/1 under LNC=2 | `b_lnc2_program_id::probe_bare_call` |
| `program_ndim` | S | `()` | 1 under LNC=2, 0 under [1] | `b_lnc2_program_id::probe_bare_call` |
| `psum` | S | `` | private per program; 2 MiB / core | `c_nc_matmul_accumulate::bf16_k256_accumulate_explicit_fp32_out` |
| `rand` | P | `(shape, dtype, buffer=nl.sbuf, name='')` | docstring: Create a new tensor of given shape and dtype on the specified buffer, filled with random values | pkg |
| `random_seed` | P | `(seed)` | docstring: Set the random seed for random number generation | pkg |
| `range` | S | `(start, stop=None, step=1)` | alias of affine_range; unrolled | `f_dynamic_range_and_ranges::loop_carried_nl.range` |
| `reciprocal` | P | `(x, y=None, dtype=None)` | docstring: Reciprocal of the input, element-wise | pkg |
| `relu` | P | `(x, y=None, dtype=None)` | docstring: ReLU activation, element-wise | pkg |
| `right_shift` | S | `(x, y=None, dtype=None)` | logical shift on int32 | `o_int_dtypes_creation_capacity::int32_right_shift_const` |
| `rms_norm` | P | `(x, w, axis, n, epsilon=1e-06, dtype=None, compute_dtype=None)` | experimental convenience composed of nisa instructions; allocates SBUF temporaries | pkg |
| `rsqrt` | S | `(x, y=None, dtype=None)` | activation op / nl.rsqrt; there is no nisa.rsqrt | pkg |
| `sbuf` | S | `` | private per program | `b_lnc2_program_id::probe_bare_call` |
| `sequential_range` | D | `(start, stop=None, step=1)` | same as affine_range | `f_dynamic_range_and_ranges::loop_carried_sequential_range` |
| `shared_constant` | P | `(constant)` | docstring: Create a tensor in shared HBM initialized with constant data | pkg |
| `shared_hbm` | S | `` | kernel I/O; visible to both programs | `h_dma_compute_lnc_comm::core_barrier_shared_hbm` |
| `shared_identity_matrix` | P | `(n, dtype='uint8', dst: 'NkiTensor \| None' = None)` | docstring: Create an identity matrix in SBUF with the specified data type | pkg |
| `sigmoid` | P | `(x, y=None, dtype=None)` | docstring: Sigmoid activation, element-wise | pkg |
| `sign` | P | `(x, y=None, dtype=None)` | docstring: Sign of the numbers of the input, element-wise | pkg |
| `silu` | P | `(x, y=None, dtype=None)` | docstring: SiLU (Swish) activation, element-wise | pkg |
| `silu_dx` | P | `(x, y=None, dtype=None)` | docstring: Derivative of SiLU activation, element-wise | pkg |
| `sin` | P | `(x, y=None, dtype=None)` | docstring: Sine of the input, element-wise | pkg |
| `softmax` | P | `(x, axis=-1, dtype=None)` | experimental convenience composed of nisa instructions; allocates SBUF temporaries | pkg |
| `softplus` | P | `(x, y=None, dtype=None)` | docstring: Softplus activation, element-wise | pkg |
| `sqrt` | P | `(x, y=None, dtype=None)` | docstring: Non-negative square-root of the input, element-wise | pkg |
| `square` | P | `(x, y=None, dtype=None)` | docstring: Square of the input, element-wise | pkg |
| `static_range` | D | `(start, stop=None, step=1)` | same as affine_range | `f_dynamic_range_and_ranges::loop_carried_static_range` |
| `store` | S | `(dst: 'NkiTensor', value: 'NkiTensor')` | experimental wrapper of dma_copy | `j_load_store::nl_load_store_tiled` |
| `subtract` | S | `(x, y=None, dtype=None)` |  | `r_skill_examples::skill_example_fori_rows` |
| `sum` | S | `(x, axis, dtype=None, keepdims=False)` | reduction -> new SBUF tile | `g_compute_ops::nl_convenience_ops` |
| `tan` | P | `(x, y=None, dtype=None)` | docstring: Tangent of the input, element-wise | pkg |
| `tanh` | P | `(x, y=None, dtype=None)` | docstring: Hyperbolic tangent, element-wise | pkg |
| `tfloat32` | P | `` | docstring: str(object='') -> str str(bytes_or_buffer[, encoding[, errors]]) -> str | pkg |
| `tile_size` | S | `` | see SKILL 3.2; version-dependent properties only inside a kernel; `psum_fmax_bytes` absent | `l_python_subset::tile_size_properties_in_kernel` |
| `tiles` | P | `(source: nki.language.tensor.NkiTensor \| nki.language._grid.grid.NkiTileGrid \| nki.language._grid.grid.NkiBlockGrid, *, tile_shape: tuple[int, ...]  ...` | new in 0.7.0b1 (tiled-grid helpers); not covered by SKILL.md | pkg |
| `transpose` | R | `(x, dtype=None)` | PSUM result in input dtype (pkg); not device-tested | pkg |
| `trunc` | P | `(x, y=None, dtype=None)` | docstring: Truncated value of the input, element-wise | pkg |
| `uint16` | P | `` | docstring: str(object='') -> str str(bytes_or_buffer[, encoding[, errors]]) -> str | pkg |
| `uint32` | P | `` | docstring: str(object='') -> str str(bytes_or_buffer[, encoding[, errors]]) -> str | pkg |
| `uint8` | S | `` | torch.bool maps here | `o_int_dtypes_creation_capacity::torch_bool_input` |
| `var` | P | `(x, axis, dtype=None, keepdims=False)` | experimental convenience composed of nisa instructions; allocates SBUF temporaries | pkg |
| `where` | P | `(condition, x, y, dtype=None)` | experimental convenience composed of nisa instructions; allocates SBUF temporaries | pkg |
| `while_loop` | S | `(init, body_fun)` | init is a VirtualRegister; body returns the next condition register | `e_fori_loop_dynamic_offset::while_loop_countdown` |
| `zeros` | R | `(shape, dtype, buffer=nl.sbuf, name='')` | SBUF default; not allowed in shared_hbm | `o_int_dtypes_creation_capacity::creation_helpers`, `o_int_dtypes_creation_capacity::zeros_in_shared_hbm` |
| `zeros_like` | S | `(x, dtype=None, buffer=None, name='')` |  | `o_int_dtypes_creation_capacity::creation_helpers` |

### `nki.isa`

| Name | Status | Signature (installed) | Note | Evidence |
|---|---|---|---|---|
| `NkiInstruction` | P | `(*args: Any, **kwargs: Any) -> Any` | docstring: Opaque handle to a traced NKI instruction | pkg |
| `NkiScalar` | P | `(*args: Any, **kwargs: Any) -> Any` | docstring: A scalar value held in a sequencer register | pkg |
| `NkiValidationError` | P | `` | docstring: Raised when hardware constraints are violated | pkg |
| `VirtualRegister` | P | `(*args: Any, **kwargs: Any) -> Any` | docstring: A virtual register on engine | pkg |
| `activate2` | G4 | `(dst: 'NkiTensor', op, data: 'NkiTensor', imm0: 'float \| NkiTensor', imm1: 'float \| NkiTensor', op0, op1, relu_param: 'float \| NkiTensor' = 0.0, re ...` |  | `n_gen4_only_and_fp8_perf_mode::activate2_gen4_only` |
| `activation` | R | `(dst: 'NkiTensor', op, data: 'NkiTensor', bias: 'float \| NkiTensor \| None' = None, scale: 'float \| NkiTensor' = 1.0, reduce_op=None, reduce_res: 'N ...` | ScalarE; reduce_op=nl.add only; engine=gpsimd is v4-only | `g_compute_ops::activation_bias_scale_fused_reduce`, `g_compute_ops::activation_engine_gpsimd` |
| `activation_reduce` | S | `(dst: 'NkiTensor', op, data: 'NkiTensor', reduce_op, reduce_res: 'NkiTensor', bias: 'float \| NkiTensor \| None' = None, scale: 'float \| NkiTensor' = ...` | legacy form of activation(reduce_cmd=reset_reduce); not device-tested | pkg |
| `affine_select` | P | `(dst: 'NkiTensor', pattern, channel_multiplier, on_true_tile: 'NkiTensor', on_false_value, cmp_op=equal, offset=0, name=None)` | docstring: Select elements between an input tile `on_true_tile` and a scalar value `on_false_value` according to a boolean predi... | pkg |
| `bn_aggr` | P | `(dst: 'NkiTensor', data: 'NkiTensor', name=None)` | docstring: Aggregate one or multiple `bn_stats` outputs to generate a mean and variance per partition using Vector Engine | pkg |
| `bn_stats` | P | `(dst: 'NkiTensor', data: 'NkiTensor', name=None)` | docstring: Compute mean- and variance-related statistics for each partition of an input tile `data` in parallel using Vector Engine | pkg |
| `core_barrier` | S | `(data: 'NkiTensor', cores, engine=<engine.gpsimd: 3>, name=None)` | cores=(0, 1) only; data in shared HBM | `h_dma_compute_lnc_comm::core_barrier_shared_hbm` |
| `dge_mode` | S | `(*values)` | unknown/swdge/hwdge/none | `m_data_movement::dma_transpose_bf16_hwdge_16rows` |
| `dma_compute` | R | `(dst: 'NkiTensor', srcs: 'list[NkiTensor]', reduce_op, scales=None, unique_indices=True, oob_mode=<oob_mode.error: 0>, name=None)` | reduce_op=nl.add only; scales all 1.0 | `h_dma_compute_lnc_comm::dma_compute_add_sbuf`, `h_dma_compute_lnc_comm::dma_compute_add_hbm`, `h_dma_compute_lnc_comm::dma_compute_multiply` |
| `dma_copy` | R | `(dst: 'NkiTensor', src: 'NkiTensor', priority=None, oob_mode=<oob_mode.error: 0>, dge_mode=<dge_mode.unknown: 0>, engine=engine.unknown, name=None)` | HBM/SBUF only (no PSUM); dtype conversion; priority= is v4-only | `a_elementwise_add_launch::bare_call`, `m_data_movement::dma_copy_psum_to_hbm`, `i_dtype_cast::dma_copy_fp32_sbuf_to_bf16_hbm`, `n_gen4_only_and_fp8_perf_mode::dma_copy_priority_gen4_only` |
| `dma_engine` | P | `(*values)` | docstring: DMA transfer engine | pkg |
| `dma_transpose` | R | `(dst: 'NkiTensor', src: 'NkiTensor', axes=None, priority=None, dge_mode=<dge_mode.unknown: 0>, oob_mode=<oob_mode.error: 0>, name=None)` | default dge_mode: bf16/fp32 OK; hwdge: 2-byte dtype, 2-D src.shape[0]==16 | `m_data_movement::dma_transpose_fp32_default`, `m_data_movement::dma_transpose_fp32_hwdge`, `m_data_movement::dma_transpose_bf16_hwdge_16rows` |
| `dropout` | P | `(dst: 'NkiTensor', data: 'NkiTensor', prob: 'float \| NkiTensor', name=None)` | docstring: Randomly replace some elements of the input tile `data` with zeros based on input probabilities using Vector Engine | pkg |
| `engine` | S | `(*values)` | tensor/vector/scalar/gpsimd/dma/sync/unknown | `g_compute_ops::engine_selection` |
| `exponential` | G4 | `(dst: 'NkiTensor', src: 'NkiTensor', max_value: 'float \| NkiTensor' = 0.0, reduce_res: 'NkiTensor \| None' = None, reduce_cmd: 'reduce_cmd_enum' = <r ...` |  | `n_gen4_only_and_fp8_perf_mode::exponential_gen4_only` |
| `get_nc_sub_version` | P | `()` | docstring: Returns the sub-version integer for the current target context | pkg |
| `get_nc_version` | P | `()` | docstring: Returns the nc_version of the current target context | pkg |
| `gpsimd_engine` | P | `` |  | pkg |
| `iota` | S | `(dst: 'NkiTensor', pattern, offset=0, channel_multiplier=0, name=None)` | int32 pattern, SBUF only | `g_compute_ops::iota_cast_reciprocal_scan` |
| `local_gather` | P | `(dst: 'NkiTensor', src_buffer: 'NkiTensor', index: 'NkiTensor', num_elem_per_idx=1, num_valid_indices=None, name=None)` | docstring: Gather SBUF data in `src_buffer` using `index` on GpSimd Engine | pkg |
| `matmul_perf_mode` | R | `(*values)` | double_row: fp8 operands [128, 2, X] | `n_gen4_only_and_fp8_perf_mode::nc_matmul_double_row_fp8_e4m3`, `n_gen4_only_and_fp8_perf_mode::nc_matmul_double_row_bf16` |
| `max8` | P | `(dst: 'NkiTensor', src: 'NkiTensor', name=None)` | docstring: Find the 8 largest values in each partition of the source tile | pkg |
| `memset` | S | `(dst: 'NkiTensor', value, engine=engine.unknown, name=None)` | constant fill; integral float into int32 stored as int | `l_python_subset::memset_float_value_into_int32_tile` |
| `nc_find_index8` | P | `(dst: 'NkiTensor', data: 'NkiTensor', vals: 'NkiTensor', name=None)` | docstring: Find indices of the 8 given vals in each partition of the data tensor | pkg |
| `nc_match_replace8` | P | `(dst: 'NkiTensor', data: 'NkiTensor', vals: 'NkiTensor', imm: 'float', dst_idx: 'NkiTensor \| None' = None, name=None)` | docstring: Replace first occurrence of each value in `vals` with `imm` in `data` using the Vector engine and return the replaced... | pkg |
| `nc_matmul` | R | `(dst: 'NkiTensor', stationary: 'NkiTensor', moving: 'NkiTensor', is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, accumulate= ...` | SBUF operands, fp32 PSUM dst, K,M<=128, N<=512; no int8/e4m3fn; fp32 x bf16 rejected | `c_nc_matmul_accumulate::bf16_k256_accumulate_explicit_fp32_out`, `d_negative_limits::psum_matmul_n_1024`, `d_negative_limits::nc_matmul_int8` |
| `nc_matmul_mx` | G4 | `(dst: 'NkiTensor', stationary: 'NkiTensor', moving: 'NkiTensor', stationary_scale: 'NkiTensor', moving_scale: 'NkiTensor', tile_position=None, tile_si ...` | docstring: NeuronCore-v4+; not device-tested | pkg |
| `nc_n_gather` | P | `(dst: 'NkiTensor', data: 'NkiTensor', indices: 'NkiTensor', name=None)` | docstring: Gather elements from `data` according to `indices` using GpSimd Engine | pkg |
| `nc_stream_shuffle` | P | `(dst: 'NkiTensor', src: 'NkiTensor', shuffle_mask, name=None)` | docstring: Apply cross-partition data movement within a quadrant of 32 partitions from source tile `src` to destination tile `ds... | pkg |
| `nc_transpose` | R | `(dst: 'NkiTensor', data: 'NkiTensor', engine=engine.unknown, name=None)` | TensorE <=128x128, PSUM dst dtype == input; VectorE <=32x32 | `g_compute_ops::nc_transpose_tensor_engine_bf16`, `g_compute_ops::nc_transpose_tensor_engine_fp32_psum_for_bf16`, `g_compute_ops::nc_transpose_vector_engine_64x64` |
| `nc_version` | P | `(*values)` | docstring: NeuronCore version | pkg |
| `nonzero_with_count` | P | `(dst: 'NkiTensor', src: 'NkiTensor', index_offset=0, padding_val=-1, name=None)` | docstring: Find indices of nonzero elements in an input tensor and their total count using GpSimd Engine | pkg |
| `oob_mode` | P | `(*values)` | docstring: Out-of-bounds access mode | pkg |
| `quantize_mx` | G4 | `(dst: 'NkiTensor', src: 'NkiTensor', dst_scale: 'NkiTensor', name=None)` | rejected by neuronx-cc (NCC_IBIR530) | `n_gen4_only_and_fp8_perf_mode::quantize_mx_gen4_only` |
| `rand2` | G4 | `(dst: 'NkiTensor', min: 'float \| NkiTensor', max: 'float \| NkiTensor', name=None)` | docstring: NeuronCore-v4+ | pkg |
| `rand_get_state` | P | `(dst: 'NkiTensor', engine=<engine.gpsimd: 3>, name=None)` | docstring: Store the current pseudo random number generator (PRNG) states from the engine | pkg |
| `rand_set_state` | P | `(src_seeds: 'NkiTensor', engine=<engine.gpsimd: 3>, name=None)` | docstring: Seed the pseudo random number generator (PRNG) inside the engine | pkg |
| `range_select` | P | `(dst: 'NkiTensor', on_true_tile: 'NkiTensor', comp_op0, comp_op1, bound0: 'NkiTensor', bound1: 'NkiTensor', reduce_cmd=<reduce_cmd.reset_reduce: 3>, r ...` | docstring: Select elements from `on_true_tile` based on comparison with bounds using Vector Engine | pkg |
| `reciprocal` | S | `(dst: 'NkiTensor', data: 'NkiTensor', name=None)` |  | `g_compute_ops::iota_cast_reciprocal_scan` |
| `reduce_cmd` | S | `(*values)` | idle/reset/reduce/reset_reduce/load_reduce | `g_compute_ops::activation_bias_scale_fused_reduce` |
| `register_alloc` | S | `(x: 'int \| None' = None)` |  | `e_fori_loop_dynamic_offset::fori_loop_register_upper_bound` |
| `register_load` | S | `(dst, src: 'NkiTensor')` |  | `e_fori_loop_dynamic_offset::while_loop_countdown` |
| `register_move` | P | `(dst, src)` | docstring: Move a value into a virtual register | pkg |
| `register_store` | P | `(dst: 'NkiTensor', src)` | docstring: Store the value from a virtual register into memory (HBM/SBUF) | pkg |
| `rng` | P | `(dst: 'NkiTensor', engine=<engine.vector: 5>, name=None)` | docstring: Generate pseudo random numbers using the Vector or GpSimd Engine | pkg |
| `scalar_engine` | P | `` |  | pkg |
| `scalar_load` | P | `(src: 'NkiTensor') -> 'NkiScalar'` | docstring: Load a scalar value from a 1x1 tensor into a sequencer register | pkg |
| `scalar_store` | P | `(dst: 'NkiTensor', value: 'NkiScalar \| int') -> 'None'` | docstring: Store a scalar value from a register to a 1x1 tensor in memory | pkg |
| `scalar_tensor_tensor` | P | `(dst: 'NkiTensor', data: 'NkiTensor', op0, operand0: 'float \| NkiTensor', op1, operand1: 'NkiTensor', reverse0=False, reverse1=False, reduce_cmd: 're ...` | docstring: Apply two math operators in sequence using Vector Engine: `(data <op0> operand0) <op1> operand1` | pkg |
| `select_reduce` | P | `(dst: 'NkiTensor', predicate: 'NkiTensor', on_true: 'NkiTensor', on_false: 'float \| NkiTensor', reduce_res: 'NkiTensor \| None' = None, reduce_cmd=<r ...` | docstring: Selectively copy elements from either `on_true` or `on_false` to the destination tile based on a `predicate` using Ve... | pkg |
| `sendrecv` | S | `(src: 'NkiTensor', dst: 'NkiTensor', send_to_rank, recv_from_rank, pipe_id, dma_engine=<dma_engine.dma: 1>, name=None)` | SBUF src/dst; ranks 0/1 | `h_dma_compute_lnc_comm::sendrecv_swap` |
| `sendrecv_compute` | S | `(src: 'NkiTensor', dst: 'NkiTensor', send_to_rank, recv_from_rank, reduce_op, name=None)` | reduce_op add/max/min | `h_dma_compute_lnc_comm::sendrecv_compute_add` |
| `sequence_bounds` | P | `(dst: 'NkiTensor', segment_ids: 'NkiTensor', name=None)` | docstring: Compute the sequence bounds for a given set of segment IDs using GpSIMD Engine | pkg |
| `set_rng_seed` | P | `(src_seeds: 'NkiTensor', name=None)` | docstring: Seed the pseudo random number generator (PRNG) inside the Vector Engine | pkg |
| `tensor_copy` | S | `(dst: 'NkiTensor', src: 'NkiTensor', engine=engine.unknown, name=None)` | PSUM->SBUF, casts (RNE) | `i_dtype_cast::tensor_copy_fp32_to_bf16` |
| `tensor_copy_predicated` | P | `(dst: 'NkiTensor', src: 'float \| NkiTensor', predicate: 'NkiTensor', reverse_pred=False, name=None)` | docstring: Conditionally copy elements from the `src` tile to the `dst` tile on SBUF / PSUM based on a `predicate` using Vector ... | pkg |
| `tensor_engine` | P | `` |  | pkg |
| `tensor_partition_reduce` | P | `(dst: 'NkiTensor', op, data: 'NkiTensor', name=None)` | docstring: Apply a reduction operation across partitions of an input `data` tile using GpSimd Engine | pkg |
| `tensor_reduce` | S | `(dst: 'NkiTensor', op, data: 'NkiTensor', axis, negate=False, keepdims=False, name=None)` | last contiguous free axes; negate= works | `g_compute_ops::tensor_reduce_free_axes` |
| `tensor_scalar` | R | `(dst: 'NkiTensor', data: 'NkiTensor', op0, operand0: 'float \| NkiTensor', reverse0=False, op1=None, operand1: 'float \| NkiTensor \| None' = None, re ...` | tile operands must be fp32 for arithmetic ops | `g_compute_ops::tensor_scalar_vector_operand_fp32`, `g_compute_ops::tensor_scalar_vector_operand_fp16` |
| `tensor_scalar_cumulative` | P | `(dst: 'NkiTensor', src: 'NkiTensor', op0, op1, imm0: 'float \| NkiTensor', imm1: 'float \| NkiTensor \| None' = None, reduce_cmd=<reduce_cmd.reset_red ...` | docstring: Perform tensor-scalar arithmetic operation with cumulative reduction using Vector Engine | pkg |
| `tensor_scalar_reduce` | P | `(dst: 'NkiTensor', data: 'NkiTensor', op0, operand0: 'float \| NkiTensor', reduce_op, reduce_res: 'NkiTensor \| None' = None, reverse0=False, reduce_c ...` | docstring: Perform the same computation as `nisa.tensor_scalar` with one math operator and also a reduction along the free dimen... | pkg |
| `tensor_tensor` | R | `(dst: 'NkiTensor', data1: 'NkiTensor', data2: 'NkiTensor', op, engine=engine.unknown, name=None)` | not both inputs in PSUM; dst SBUF/PSUM; int32 exact on GpSimd only | `g_compute_ops::engine_selection`, `g_compute_ops::tensor_tensor_both_inputs_psum`, `o_int_dtypes_creation_capacity::int32_add_vector_engine` |
| `tensor_tensor_scan` | S | `(dst: 'NkiTensor', data0: 'NkiTensor', data1: 'NkiTensor', initial: 'float \| NkiTensor', op0, op1, reverse0=False, reverse1=False, name=None)` |  | `g_compute_ops::iota_cast_reciprocal_scan` |
| `topk` | R | `(val_dst: 'NkiTensor', idx_dst: 'NkiTensor', src: 'NkiTensor', n, sorted=False, name=None)` | v3+; BF16 snake layout per 16-partition group; minimal call ran | `n_gen4_only_and_fp8_perf_mode::topk_available_trn2` |
| `unknown_engine` | P | `` |  | pkg |
| `vector_engine` | P | `` |  | pkg |

### `nki`

| Name | Status | Signature (installed) | Note | Evidence |
|---|---|---|---|---|
| `Kernel` | S | `(func: Callable, lnc: int \| None = None, schedule: tuple = (), address_rotation: nki.framework.kernel.AddressRotation = <AddressRotation.AUTO: 'auto' ...` | `kernel[1]` / `kernel[2]` select LNC; unset LNC resolves to NEURON_LOGICAL_NC_CONFIG | `b_lnc2_program_id::probe_lnc1_bracket` |
| `debug` | P | `(kernel, dump_dir, core_id=0, interactive=False, rtol=0.01, atol=0.1)` |  | pkg |
| `dtype` | P | `(*args, **kwargs)` | type alias | pkg |
| `jit` | S | `(fn=None, *, compile_config=None, specialization_config=None, **kwargs)` | `jit(fn=None, *, compile_config=None, specialization_config=None, **kwargs)`; lnc=/platform_target= rejected, mode= ignored | `a_elementwise_add_launch::bare_call` |
| `simulate` | P | `(kernel)` | CPU simulator entry point | pkg |

### `nki.compiler`

| Name | Status | Signature (installed) | Note | Evidence |
|---|---|---|---|---|
| `AddressRotation` | P | `(*values)` | per-kernel config objects | pkg |
| `BatchCompilationSummary` | P | `(total_configs: 'int', successful: 'int', failed: 'int', total_time: 'float', results: 'List[CompilationResult]') -> None` | internal per module docstring | pkg |
| `CompilationResult` | P | `(config_id: 'int', config_label: 'str', success: 'bool', neff_path: 'Optional[str]' = None, compilation_time: 'float' = 0.0, mlir_time: 'float' = 0.0, ...` | internal per module docstring | pkg |
| `CompileConfig` | P | `(address_rotation: nki.framework.kernel.AddressRotation = <AddressRotation.AUTO: 'auto'>, non_ssa_legalization: bool = False, dead_code_elim: bool = T ...` | per-kernel config objects | pkg |
| `KernelSpec` | P | `(kernel_module: 'str', kernel_func_name: 'str', inputs: 'Dict[str, np.ndarray]', outputs: 'Dict[str, np.ndarray]', hyperparams: 'Dict[str, Any]', conf ...` | internal per module docstring | pkg |
| `SpecializationConfig` | P | `(allow_nested_config_override: bool = False) -> None` | per-kernel config objects | pkg |
| `get_compiler_version` | P | `() -> nki.compiler._version.CompilerVersion` | internal per module docstring | pkg |
| `parallel_compile` | P | `(kernel_specs: 'List[KernelSpec]', compile_opts_dict: 'Dict[str, Any]', artifacts_base_dir: 'str', max_workers: 'int' = 8, progress_callback: 'Optiona ...` | internal per module docstring | pkg |

### `nki.collectives`

| Name | Status | Signature (installed) | Note | Evidence |
|---|---|---|---|---|
| `ReplicaGroup` | P | `(*args: Any, **kwargs: Any) -> Any` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `all_gather` | P | `(srcs: 'List[NkiTensor]', dsts: 'List[NkiTensor]', replica_group: 'ReplicaGroup', collective_dim: 'int', priority: 'Optional[int]' = None, name: 'Opti ...` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `all_gather_v` | P | `(srcs: 'List[NkiTensor]', dsts: 'List[NkiTensor]', replica_group: 'ReplicaGroup', metadata_tensor: 'NkiTensor', recv_counts_known: 'bool' = False, has ...` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `all_gather_v_prepare` | P | `(srcs: 'List', dsts: 'List', replica_group: 'ReplicaGroup', metadata_tensor, recv_counts_known: 'bool' = False, has_rdispls: 'bool' = False, priority: ...` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `all_gather_v_trigger` | P | `(prepare_instr: 'NkiInstruction', name: 'Optional[str]' = None) -> 'NkiInstruction'` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `all_reduce` | P | `(srcs: 'List[NkiTensor]', dsts: 'List[NkiTensor]', replica_group: 'ReplicaGroup', op, priority: 'Optional[int]' = None, name: 'Optional[str]' = None)  ...` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `all_to_all` | P | `(srcs: 'List[NkiTensor]', dsts: 'List[NkiTensor]', replica_group: 'ReplicaGroup', collective_dim: 'int', priority: 'Optional[int]' = None, name: 'Opti ...` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `all_to_all_v` | P | `(srcs: 'List[NkiTensor]', dsts: 'List[NkiTensor]', replica_group: 'ReplicaGroup', metadata_tensor: 'NkiTensor', recv_counts_known: 'bool' = False, has ...` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `all_to_all_v_prepare` | P | `(srcs: 'List', dsts: 'List', replica_group: 'ReplicaGroup', metadata_tensor, recv_counts_known: 'bool' = False, has_rdispls: 'bool' = False, priority: ...` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `all_to_all_v_trigger` | P | `(prepare_instr: 'NkiInstruction', name: 'Optional[str]' = None) -> 'NkiInstruction'` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `collective_permute` | P | `(srcs: 'List[NkiTensor]', dsts: 'List[NkiTensor]', source_target_pairs: 'List[Tuple[int, int]]', priority: 'Optional[int]' = None, name: 'Optional[str ...` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `collective_permute_implicit` | P | `(srcs_by_channel: 'List[List[NkiTensor]]', dsts_by_channel: 'List[List[NkiTensor]]', replica_group: 'ReplicaGroup', channel_ids: 'List[int]' = [0], pr ...` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `collective_permute_implicit_current_processing_rank_id` | P | `(iteration_id: 'int', replica_group: 'ReplicaGroup', channel_id: 'int' = 0, name: 'Optional[str]' = None)` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `collective_permute_implicit_reduce` | P | `(srcs0_by_channel: 'List[List[NkiTensor]]', srcs1_by_channel: 'List[List[NkiTensor]]', dsts_by_channel: 'List[List[NkiTensor]]', replica_group: 'Repli ...` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `rank_id` | P | `(name: 'Optional[str]' = None)` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `reduce_scatter` | P | `(srcs: 'List[NkiTensor]', dsts: 'List[NkiTensor]', replica_group: 'ReplicaGroup', collective_dim: 'int', op, priority: 'Optional[int]' = None, name: ' ...` | multi-rank collectives; not needed for single-logical-core kernels | pkg |
| `rg_partition_op` | P | `(*values)` | multi-rank collectives; not needed for single-logical-core kernels | pkg |

### Names used by older public material that are absent or rejected in 0.7.0b1

| Module | Name | Replacement / note | Evidence |
|---|---|---|---|
| `nki.language` | `arange` | Beta-1 index generator | `d_negative_limits::removed_nl_arange` |
| `nki.language` | `mgrid` | Beta-1 index grid | introspection `legacy_names_present` |
| `nki.language` | `par_dim` | partition-dim marker | `d_negative_limits::removed_nl_par_dim` |
| `nki.language` | `atomic_rmw` | use nisa.dma_compute | introspection |
| `nki.language` | `spmd_dim / nc` | old LNC launch grid (`kernel[nl.spmd_dim(n, nl.nc(2))]`) | introspection |
| `nki.language` | `loop_reduce, static_cast, mask` | Beta-1 helpers | introspection |
| `nki.language` | ``mask=` keyword` | Beta-1 masking | `d_negative_limits::removed_mask_kwarg` |
| `nki.language` | `tile_size.psum_fmax_bytes` | use psum_bank_fmax_bytes | pkg `_tile_size.py` |
| `nki.isa` | `rsqrt` | use activation(op=nl.rsqrt) | introspection |
| `nki.isa` | `tensor_copy_dynamic_src / _dst` | use .ap(scalar_offset=...) | introspection |
| `nki.isa` | `builtin_custom_op` |  | introspection |
| `nki.isa` | `activate` |  | introspection |
| `nki.isa` | `constants.engine` | nki.isa.constants exports only dge_mode, oob_mode, matmul_perf_mode | pkg `isa/constants.py` |
| `nki` | `benchmark, baremetal, profile, trace, simulate_kernel, tensor` | older top-level helpers | introspection |
| `nki.jit` | `lnc=, platform_target=` | TypeError / ValueError | pkg `nki/__init__.py` |
| `framework` | `torch_xla / xm.xla_device()` | not installed in this stack | introspection `versions.torch_xla_importable` |
