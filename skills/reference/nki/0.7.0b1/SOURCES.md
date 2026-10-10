# Sources for the NKI 0.7.0b1 Reference Skill

`SKILL.md` ("NKI 0.7.0b1 Reference for AWS Trainium2") and `COMPATIBILITY.md` were written from the sources below only. Priority order: (1) installed package, (2) neuron-agentic-development, (3) nki-samples (licence and API style check only), (4) AWS Neuron documentation. Every API name, signature and default in `SKILL.md` was taken from source (1) or confirmed against it. Every behavioural claim tagged `[dev:x]` was observed in `evidence/smoke_tests/x_*.py` on the trn2 host. All sources were accessed on **2026-10-10**.

## 1. Installed package (ground truth)

| Item | Value |
|---|---|
| Interpreter | `/home/ubuntu/workspace/native_venv/bin/python` (Python 3.12.3) |
| `nki` | `0.7.0b1+34883233359.g2aa6a82a` (pre-release, "Beta 5"). Installed from the local wheel `file:///home/ubuntu/workspace/nki_wheels/nki-0.7.0b1+34883233359.g2aa6a82a-cp312-cp312-linux_x86_64.whl` (`direct_url.json`). Wheel metadata: `License-Expression: Apache-2.0`, `Author-Email: AWS Neuron Team`, NOTICE "Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved."; `Project-URL: Homepage, https://github.com/aws-neuron/private-nki-staging` (a private repository). Public availability check (2026-10-10): https://pip.repos.neuron.amazonaws.com/nki/ lists versions up to `0.6.0+31049202112.g85070674`; https://pypi.org/pypi/nki/json is an unrelated project (0.0.1, 0.1) |
| `neuronx-cc` | `2.0.404830.0a0+55f4bf89` (licence: AWS Neuron Software License Agreement; only its error messages are quoted) |
| `torch` | `2.14.0+cpu` |
| `torch-neuronx` | `2.14.3.0.0+unknown.dev`, editable source `/home/ubuntu/workspace/torch_neuron_eager/torch_neuronx` (native PyTorch device `neuron`) |
| Other | numpy 2.5.3, islpy 2026.2.2, ml_dtypes 0.6.0; torch-xla not importable |
| Host | trn2.3xlarge (1 Trainium2, 8 physical NeuronCore-v3), `neuron-ls`: `logical-neuroncore-config: 2`, 4 logical cores |

Files read (paths relative to `site-packages/`):

- `nki/__init__.py` (`jit`), `nki/_config.py` (all environment options, `NKI_COMPILER_FRONTEND`), `nki/framework/kernel.py` (`Kernel`, `CompileConfig`, `SpecializationConfig`, nested-call handling, torch dispatch), `nki/framework/_lnc.py` (`resolve_lnc`), `nki/framework/compiled.py` (cache keys, trace cache), `nki/framework/torch_native.py`, `nki/framework/_torch_convert.py` (torch ↔ NKI dtype maps, `float8_e4m3fn` handling), `nki/compiler/__init__.py`, `nki/compiler/frontend.py` (`resolve_frontend_cls`).
- `nki/language/__init__.py`, `_core.py` (`ndarray`, `zeros`, `full`, `load`, `store`, `load_transpose2d`, `program_id`, `num_programs`, `program_ndim`, `rand`), `_ranges.py` (`affine_range`, `sequential_range`, `static_range`, `dynamic_range`, `fori_loop`, `while_loop`), `_tile_size.py`, `buffers.py`, `tensor.py` (`NkiTensor.__getitem__`, `__setitem__`, `reshape`, `slice`, `select`, `ap`, `view`, `ds`), `_ops.py`, `_operators.py`, `_math_funcs.py` (`matmul`, `transpose`, reductions), `opcode.py`, `_dtypes.py`.
- `nki/isa/__init__.py`, `_matmul.py`, `_copy.py`, `_tensor_ops.py`, `_activation.py`, `_transpose.py`, `_lnc.py`, `_advanced.py` (`dma_compute`, `range_select`, `select_reduce`, `nc_matmul_mx`, `quantize_mx`, registers), `_misc.py` (`topk`, `exponential`, `reciprocal`), `_scalar.py` (`NkiScalar`), `_virtual_register.py`, `_enums.py`, `constants.py`, `_validation.py` (matmul dtype/shape/buffer validation, transpose rules, activation op table, generation checks, DMA-transpose rules).
- `nki/collectives/__init__.py` (symbol list only).
- torch-neuronx: `torch_neuronx/nki_hop.py` (`NKIHOPCaller`, eager dispatch, int64 handling), `torch_neuronx/nki_kernel.py` (trace-cache warning, `get_config`).

Generated evidence: `evidence/introspection.json` (by `evidence/introspect.py`): symbol lists, `inspect.signature`, full docstrings, source locations, enum members, dtypes, `nl.tile_size` constants, torch→NKI dtype map, presence of 120 names used by older material (18 absent), `jit`/`Kernel`/`CompileConfig` fields, all `nki._config` options.

## 2. aws-neuron/neuron-agentic-development (Apache-2.0)

| Item | Value |
|---|---|
| Repository | https://github.com/aws-neuron/neuron-agentic-development |
| Commit | `ee25e45b9ace4aaefd8b8913bf5fa9b68adf6674` ("AWS Neuron 2.32; neuron-agentic-development 1.3", committed 2026-08-19) |
| Retrieval | `git clone https://github.com/aws-neuron/neuron-agentic-development` (HEAD = the commit above); compared with the verbatim local copy `/home/ubuntu/.claude/skills/nki/official/` (identical except that each upstream `SKILL.md` is renamed `GUIDE.md`) |
| Licence | `LICENSE.txt`: Apache License 2.0 (copied byte-identically as `LICENSE.apache-2.0`, SHA-256 `2465df666ec03de5d93dfe1e38f533f6137273abfeba138727dfaa42d62285c6`); `NOTICE`: "Neuron Agentic Development / Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved." (copied as `NOTICE.neuron-agentic-development`, SHA-256 `73c699c58d946ea48103debd3434c58eabb557763aaade15556961e0fd21ac53`) |

Files read and what was used:

- `skills/neuron-nki-writing/references/nki-language-constraint.md`: list of removed and renamed APIs (Beta 1 → 0.4.0), the `accumulate=(k > 0)` rule, PSUM→SBUF→HBM rule, fori/while-loop rules. Each item was re-checked against the installed package and the device. Items that are stale for 0.7.0b1 are listed in `COMPATIBILITY.md` (rows 4–12).
- `skills/neuron-nki-docs/references/reference/migration/nki-060-dynamic-loop-migration-guide.md`: `fori_loop` / `while_loop` rules (state in SBUF/HBM, read-only index, no `no_reorder`, same trip count under LNC=2). The frontend timeline is stale (`COMPATIBILITY.md` row 2).
- `skills/neuron-nki-docs/references/programming/lnc.md`: `program_id` / `num_programs` semantics, identical control flow on both cores. The default-LNC statement is stale here (row 1).
- `skills/neuron-nki-docs/references/architecture/trainium2_arch.md`: SBUF 28 MiB (128 × 224 KiB, NeuronCore-v2: 24 MiB), PSUM 2 MiB, double-FP8 matmul layout (`[128, 2, X]`, K = 256), TensorE transpose bit-accuracy, VectorE/ScalarE/GpSimd descriptions.
- `skills/neuron-nki-writing/SKILL.md`, `references/indexing-patterns.md`, `memory-patterns.md`, `kernel-template.md`, `common-patterns.md`, `transpose-and-layout.md`, `skills/neuron-nki-docs/references/programming/*.md`, `programming/tutorials/*.md`: scanned for every `nl.*` / `nisa.*` name. Names absent from 0.7.0b1 (`nl.arange`, `nl.mgrid`, `nl.par_dim`, `nl.spmd_dim`, `nl.nc`, `nisa.rsqrt`, `nisa.activate`, `nisa.constants.engine`, `nisa.tensor_copy_dynamic_*`) are recorded in `COMPATIBILITY.md`. `programming/framework_custom_op.md` uses torch-xla (not in this stack).
- `skills/neuron-nki-docs/references/reference/nki_rn.md`: release history (cross-check with item 4).

No code was copied from this repository. The `SKILL.md` code examples are the author's own test kernels (section 5).

## 3. aws-neuron/nki-samples (MIT-0)

| Item | Value |
|---|---|
| Repository | https://github.com/aws-neuron/nki-samples |
| Commit | `62c300da468b34655aa1fc98a70d807c8dad4a72` (2026-10-07, "Add all-gather ring and tensor indirection tutorials; pin LNC=1 in mxfp-matmul (#137)"), cloned to the session scratchpad |
| Licence | `LICENSE`: "MIT No Attribution", Copyright Amazon.com, Inc. or its affiliates (SHA-256 `0beac60c03ba4648df37840205ea1abba594a08200e2665454ff26ab65e9021c`). The README says "MIT-0 License" |
| NKI version used by the samples | Tutorials updated "for Neuron SDK 2.32 release (NKI 0.6.0)" (commit `ea7dfedf`, 2026-08-26). `src/nki_samples/tutorials/tensor_addition/tensor_addition_nki_kernels.py` uses the current `import nki` / `nisa.dma_copy` / `nisa.tensor_tensor` style. Its driver uses `torch_neuronx.trace(...)`, which is not part of this stack. `contributed/*` (marked "tested only against internal nightly builds") and `tutorials/attention_fwd_performance`, `tensor_indirection` still use `nl.load`/`nl.store`/`nl.arange`-era code (found by name search only) |

Used: licence, NKI version, and the API style of `tensor_addition_nki_kernels.py` (read). The element-wise add in `SKILL.md` §1.3 follows the same public tutorial pattern (dma_copy → tensor_tensor → dma_copy over 128-row tiles). It was written and tested independently (`a_elementwise_add_launch.py`). `src/nki_samples/tutorials/mxfp-matmul/mx_kernel_utils.py` was grepped for the `quantize_mx` scale-tile shape used in the negative test `n_gen4_only_and_fp8_perf_mode.py`. No other sample code was read or used. The `contributed/` kernels and the matmul, transpose and attention tutorials were not read.

## 4. AWS Neuron documentation (CC BY-SA 4.0)

Public latest: Neuron 2.32.0 / NKI 0.6.0. Pages used (local copies downloaded on 2026-10-10 from https://awsdocs-neuron.readthedocs-hosted.com/en/latest/):

| Page | Local copy | Used for |
|---|---|---|
| NKI API index, `nki`, `nki.language`, `nki.isa` pages (`/nki/api/...`) | `/home/ubuntu/paper_reports/trn2_skill_audit/docs/nki_api_*.{html,txt}` | public API list for the comparison |
| NKI release notes (`/release-notes/components/nki.html`) | `.../trn2_skill_audit/docs/release-notes_components_nki.txt` | 0.6.0 changes (`fori_loop`/`while_loop`, `dynamic_range` deprecation warning, DMA-transpose relaxations, `topk`) |
| `nki.isa.nc_matmul`, `nki.isa.tensor_scalar`, `nki.isa.activation`, `nki.isa.matmul_perf_mode` (`/nki/api/generated/...`) | `/home/ubuntu/paper_reports/trn2_theoretical_audit/docs/nki_api_generated_*.txt` | cross-check with installed docstrings |
| NKI Trainium2 architecture guide (`/nki/guides/architecture/trainium2_arch.html`) | `.../trn2_theoretical_audit/docs/nki_guides_architecture_trainium2_arch.txt` | SBUF/PSUM capacity, double FP8 |
| Logical NeuronCore configuration (`/about-neuron/arch/neuron-features/logical-neuroncore-config.html`) | `.../trn2_theoretical_audit/docs/about-neuron_arch_neuron-features_logical-neuroncore-config.txt` | LNC=2 default on Trn2, four HBM stacks each shared by two physical cores, shared address space |
| Data types (`/about-neuron/arch/neuron-features/data-types.html`) | `.../trn2_theoretical_audit/docs/about-neuron_arch_neuron-features_data-types.txt` | RNE rounding default |
| NeuronCore-v3 (`/about-neuron/arch/neuron-hardware/neuron-core-v3.html`) | `.../trn2_theoretical_audit/docs/about-neuron_arch_neuron-hardware_neuron-core-v3.txt` | engine/dtype overview (cross-check) |

The documentation licence was confirmed from the source repository https://github.com/aws-neuron/aws-neuron-sdk (master = `e27a3986fa8625d4103a7ce611ee0c85ae79979a` on 2026-10-10): `LICENSE-DOCUMENTATION` = Creative Commons Attribution-ShareAlike 4.0 International (SHA-256 `a8f88cf9ae1075eece827dc123dcc279aa8e596b32ac36d030d221feff10057d`); `LICENSE-SAMPLECODE` = MIT No Attribution (SHA-256 `ddf9b5d9dd2fded18653e0ca4c5934008d4faea1161d754493b54a23c25eac14`). The migration guide and LNC page were read in their neuron-agentic-development mirror (section 2).

## 5. Device evidence (this host)

- `evidence/smoke_tests/*.py`: 18 standalone tests (a–r) plus `_common.py` (environment exactly as the evaluator sets it: `NEURON_LOGICAL_NC_CONFIG=2`, `NEURON_CC_FLAGS="--target trn2 --lnc 2"`, `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, `NEURON_RT_NUM_CORES=1`; all caches redirected to a fresh per-run directory).
- `evidence/run_smoke.py` runs them one at a time (≤ 900 s each) and writes `evidence/smoke_results.json`: per case the API, what was checked, pass/fail, the observed error text for negative cases, and wall time.
- `evidence/smoke_tests/r_skill_examples.py` extracts every `# example:` code block of `SKILL.md` verbatim into `evidence/smoke_tests/skill_snippets/` and runs it on the device against a CPU reference.

## 6. Exclusions

Not read, copied or paraphrased: any file named `nki-guide`; TileBench's manual NKI kernels (`impl_nki.py` in any worktree or branch), their tuning parameters, tile choices or benchmark results; `/home/ubuntu/.claude/skills/nki/references/*.md` and `/home/ubuntu/.claude/skills/nki/examples/*` (not opened). Files in the shared session scratchpad that belong to other work (for example `destindex_impl_nki_fixed.py`, `moe_nki.py`, `kernels/`) were not opened. The `README.md` of the local copy `/home/ubuntu/.claude/skills/nki/official/` (a note written by the integrating session, not part of upstream) was displayed once while listing the directory. Its remarks were not used as sources: every statement in `SKILL.md` traces to sections 1–5. No benchmark operator implementation, timing or tuning number appears in `SKILL.md`. The study's own `skills/device/Trn2/2026-10-05/SKILL.md` and the model layout of `skills/reference/tilelang/0.1.11/` (`manifest.json`, `SOURCES.md`, the head of `SKILL.md`, the evidence file list) were read only to match the format.

Context disclosure: the agent harness placed the session's memory index (`MEMORY.md`) into the context. The agent also opened two memory notes about the older 0.6.0 conda environment and the device queue (`tilebench-nki-env-pins.md`, `tilebench-nki-pr-workflow.md`) to avoid disturbing other device users. No statement in `SKILL.md` depends on them.
