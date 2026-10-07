# Sources for the TileLang 0.1.11 Reference Skill

This directory contains `SKILL.md` ("TileLang 0.1.11 Language Reference") and the evidence used to write and check it. The reference was written from the three source classes below only.

## 1. Upstream repository (primary source)

| Item | Value |
|------|-------|
| Repository | https://github.com/tile-ai/tilelang |
| Tag | `v0.1.11` |
| Commit | `cd37ed5fc35ae7a60a1277c8eb49028174ac51e6` ("[Release] Bump version to 0.1.11 (#2354)", 2026-06-08) |
| Retrieval | `git clone --depth 1 --branch v0.1.11 https://github.com/tile-ai/tilelang tilelang_src`; `git rev-parse HEAD` returned the commit above |
| `VERSION` file | `0.1.11` |
| Submodule pins (from `git ls-tree HEAD 3rdparty/`) | `3rdparty/tvm` = `ec7f7bd92c185d8c0b84a5dc17709a0860aa5714` (https://github.com/TileLang/tvm), `3rdparty/cutlass` = `b2dd65dc864e09688245b316ac46c4a6cd07e15c`, `3rdparty/composable_kernel` = `b38bb492a1a55b5abb0c345962143c0f9c482cfb`. Submodules were not cloned; the few TVM-provided symbols (`T.max`, `T.min`, `T.where`) were inspected in the installed wheel, which bundles `tilelang/3rdparty/tvm/python`. |

### License

`LICENSE` at the tag is the MIT License. Copyright line, verbatim:

> Copyright (c) Tile-AI.

The file also states: "During the period from December 1, 2024, to Mar 14, 2025, this project is subject to additional collaboration terms with Microsoft Corporation." `LICENSE.tilelang` in this directory is a byte-identical copy (SHA-256 `f70e5cdc24c3797af741930440086f9a0ad36930b05a3dfc25b420936dc2cfa1`, same as the checkout's `LICENSE`). Code snippets in `SKILL.md` that follow upstream examples (the tiled GEMM of `examples/gemm/example_gemm.py` / `README.md`, the TCGEN5MMA GEMM of `examples/gemm_sm100/`) are used under this license.

### Files read in the checkout

Repository root: `LICENSE`, `VERSION`, `README.md`, `.gitmodules`.

`docs/`:
- `docs/programming_guides/overview.md`
- `docs/programming_guides/language_basics.md`
- `docs/programming_guides/type_system.md`
- `docs/programming_guides/control_flow.md`
- `docs/programming_guides/instructions.md`
- `docs/programming_guides/python_compatibility.md`
- `docs/programming_guides/software_pipeline.md`
- `docs/programming_guides/autotuning.md` (introduction only)
- `docs/get_started/targets.md`
- `docs/compiler_internals/tensor_checks.md`
- `docs/deeplearning_operators/elementwise.md` (first sections)
- `docs/tutorials/debug_tools_for_tilelang.md` (`T.print` section)

`examples/`:
- `examples/elementwise/example_elementwise_add.py`
- `examples/gemm/example_gemm.py`
- `examples/gemm_sm100/README.md`
- `examples/gemm_sm100/gemm_tcgen5mma.py`
- `examples/norm/rms_norm.py`
- `examples/dynamic_shape/example_dynamic.py`
- one-line `grep` hits for `annotate_layout` / `use_swizzle` / `make_swizzled_layout` in `examples/` (e.g. `examples/deepseek_mhc/example_mhc_pre.py`, `examples/deepseek_mla/README.md`)

`tilelang/` (Python sources):
- `tilelang/__init__.py`, `tilelang/env.py`
- `tilelang/jit/__init__.py`, `tilelang/jit/kernel.py`, `tilelang/jit/execution_backend.py`, `tilelang/jit/adapter/tvm_ffi.py`, `tilelang/jit/adapter/base.py`
- `tilelang/language/__init__.py`, `allocate.py`, `copy_op.py`, `fill_op.py`, `loop.py`, `kernel.py`, `symbolics.py`, `reduce_op.py`, `scan_op.py`, `gemm_op.py`, `customize.py`, `atomic.py`, `math_intrinsics.py`, `annotations.py`, `print_op.py`, `proxy.py`, `dtypes.py`
- `tilelang/language/eager/__init__.py`, `tilelang/language/eager/builder.py`
- `tilelang/language/tir/__init__.py`, `tilelang/language/tir/ir.py`
- `tilelang/layout/__init__.py`, `tilelang/layout/swizzle.py`
- `tilelang/tileop/base.py`, `tilelang/tileop/gemm/registry.py`
- `tilelang/transform/pass_config.py`
- `tilelang/autotuner/capture.py`, `tilelang/autotuner/tuner.py` (symbol names only)
- `tilelang/cuda/op/gemm/gemm_mma.py`, `tilelang/cuda/intrinsics/macro/mma_macro_generator.py`

`src/` (C++ sources):
- `src/transform/legalize_safe_memory_access.cc` (out-of-bounds guards, safe value = zero)
- `src/op/gemm.cc`, `src/cuda/op/gemm.cc` (GEMM instruction selection, warp partition checks)
- `src/op/tcgen5_meta.h` (TCGEN5MMA shape/dtype table)
- `src/backend/common/target_utils.cc`, `src/backend/common/target_utils.h` (architecture predicates)

## 2. Installed package (introspection and execution)

| Item | Value |
|------|-------|
| Interpreter | `/projects/kzhou6/bcui2/env_software/miniconda3/envs/tilebench_env/bin/python` (Python 3.10), always run as `env -u LD_LIBRARY_PATH <python> ...` |
| `tilelang` | 0.1.11 |
| `apache-tvm-ffi` | 0.1.11 |
| `torch` | 2.10.0+cu130 |
| GPU | NVIDIA B200, compute capability 10.0 |

- `introspect.py` resolves every `T.*` / `tilelang.*` name in `SKILL.md`, records module, `inspect.signature` and source location, and checks that the installed source line equals the line at the same path in the v0.1.11 checkout. Output: `introspection_evidence.json` (201 present symbols, 0 missing, 0 line mismatches; 16 documented-absent names confirmed absent).
- `smoke_tests.py` extracts the snippets verbatim from `SKILL.md` (written to `smoke_snippets/`), compiles and runs them on the GPU with `TILELANG_CACHE_DIR=./tilelang_cache`, compares with PyTorch (`torch.testing.assert_close`), and checks behavioural statements of the text (current-stream launch, host-side argument checks, zero fill of out-of-bounds loads, integer division, kernel-scope statements, TF32 for fp32 GEMM, postponed annotations). Output: `smoke_results.json`.
- Generated CUDA source was inspected with `JITKernel.get_kernel_source()` to confirm the instruction selected by `T.gemm` on sm_100 (`tl::mma_sync` for a fragment accumulator, `tcgen05` for a Tensor Memory accumulator, `kTensorFloat32` for fp32 operands).

## 3. Public web documentation

No web pages were fetched. The `docs/` directory of the tagged repository (the source of https://tile-ai.github.io/tilelang) was used instead.

## 4. Style references

To match scope, depth and section style only (no content was taken), the following two files were read:
- `/projects/kzhou6/bcui2/research/tilebench/llm_wt/skills/reference/triton/3.6.0/SKILL.md`
- `/projects/kzhou6/bcui2/research/tilebench/llm_wt/skills/reference/cutile/1.5.0/SKILL.md`

## 5. Exclusions

No private or internal TileLang guide was read: nothing under `~/.claude/` (including `skills/tilelang-guide`) and nothing under `llm_wt/skills/reference/tilelang/` or `llm_b200_wt/skills/reference/tilelang/` was opened, listed, searched or copied. No TileBench benchmark solution was read: no `impl_tilelang.py` (or any other file under `/projects/kzhou6/bcui2/research/tilebench/` apart from the two style references above) was opened. No file in any TileBench worktree was modified, and no LLM API was called.

Context disclosure: the agent harness automatically placed the TileBench repository's `CLAUDE.md` and a project-memory index into the agent's session context before the work started; these were not opened by the agent and contain a few one-line remarks about TileLang usage. No statement in `SKILL.md` depends on them: each one is traced to the files listed in section 1, to introspection, or to a test in `smoke_tests.py`.
