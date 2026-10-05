# Skill / reference-document inventory for TileBench++ LLM v2

Generated 2026-10-05T03:30:53 from worktree `/projects/kzhou6/bcui2/research/tilebench/llm_wt` (branch `exp/llm`, HEAD `ea04fb36`). Read-only inventory; the two files under `docs/llm_v2/` are the only outputs. Hashes are sha256 of raw bytes. "Installed in this Claude Code session" (what `/skill` can load now) is recorded separately from "what the framework will inject", which is decided only by manifests pinning path + sha256.

Section classes used below: **API/semantics** | **hardware spec** | **tuning recommendation** | **benchmark/autotune workflow** | **example kernel** | **other**.

## 1. Repo skills (`llm_wt/skills/`, tracked, used by the legacy loader)

| path | bytes | lines | sha256[:16] | mtime | frontmatter name | notes |
|---|---|---|---|---|---|---|
| `llm_wt/skills/triton-guide/SKILL.md` | 60,502 | 1644 | `ac48c68a4d980979` | 2026-10-05T03:16:48 | triton-guide | Triton **3.6.0** claimed; `_SKILL_PATHS['triton']`; byte-identical to `.claude` copy |
| `llm_wt/skills/cutile-guide/SKILL.md` | 35,299 | 1140 | `7f25fdf978bd9f82` | 2026-10-05T03:16:48 | cutile-guide | cuTile version **unknown** (deprecated `autotune_launch` API); `_SKILL_PATHS['cutile']`; older than `.claude` copy |

Both first and last touched in commit `06320a62` (2026-09-19, PR #313); `exp/llm` has no diff vs `origin/main` under `skills/`. Frontmatter: `name`, `description`, `user-invocable: true`, `allowed-tools: Read Grep Glob`, `argument-hint`; no `version` field. The loader strips the frontmatter at runtime and pins nothing.

## 2. Project Claude skills (`Tilebench/.claude/skills/`, git-excluded via `.git/info/exclude`)

| path | bytes | lines | sha256[:16] | mtime | frontmatter name | notes |
|---|---|---|---|---|---|---|
| `Tilebench/.claude/skills/triton-guide/SKILL.md` | 60,502 | 1644 | `ac48c68a4d980979` | 2026-05-18T22:20:10 | triton-guide | identical to repo copy; mtime 2026-05-18 = origin of the public copy |
| `Tilebench/.claude/skills/cutile-guide/SKILL.md` | 36,262 | 1158 | `767f467757f89693` | 2026-07-06T04:36:01 | cutile-guide | **differs**: adds §18.3 `ct.static_iter` (see §5); mtime 2026-07-06 |
| `Tilebench/.claude/skills/calculation-guide/SKILL.md` | 20,604 | 523 | `2e88c04200950683` | 2026-05-18T22:19:49 | none | not a DSL reference (GPT-OSS profiling annotations); no frontmatter |

## 3. User Claude skills (`~/.claude/skills/`)

| path | bytes | lines | sha256[:16] | mtime | frontmatter name | notes |
|---|---|---|---|---|---|---|
| `~/.claude/skills/tilelang-guide/SKILL.md` | 13,880 | 149 | `58216107d0bb8ed0` | 2026-07-10T20:29:13 | tilelang-guide | TileLang **0.1.11** claimed (= installed); only TileLang doc anywhere |
| `~/.claude/skills/nki-guide/SKILL.md` | 13,535 | 159 | `fc04e3e1345470f3` | 2026-07-10T20:27:51 | nki-guide | **PRIVATE until confirmed**; headings/counts only; NKI beta SDK 2.20 -> nki 0.1.0/0.3.0/0.5.0 range |
| `~/.claude/skills/hopper-hpc/SKILL.md` | 18,341 | 481 | `f40456646fc7a8ee` | 2026-08-12T02:04:48 | hopper-hpc | not DSL reference (GMU cluster/Slurm) |
| `~/.claude/skills/ncu-report-skill/` (symlink -> `/home/bcui2/workspace/ncu-report-skill`, git clone of mit-han-lab/ncu-report-skill @ `1cf238d6`, MIT) | 220,934 | - | per-file in JSON | 2026-06-09 | ncu-report-skill | not DSL reference; 22 files excl. .git; includes `blackwell-cuda-programming.md` (45,341 B, Chinese CUDA C++ Blackwell guide) and `reference/08-b200-metric-names.md` |
| `~/.claude/skills/synced/7d0b14af-..._0e2d18ca-.../` | 3,970,335 | - | per-file in JSON | 2026-09-16..10-05 | (bundle) | not DSL reference; Anthropic-synced skills, 218 files: (root) (3), docs (1), docx (61), google-workspace (9), import-memory (1), morning (3), pdf (12), pptx (57), skill-creator (18), xlsx (53) |

ncu-report-skill files (path, sha256[:16]):

| path | bytes | sha256[:16] |
|---|---|---|
| `~/.claude/skills/ncu-report-skill/.gitignore` | 380 | `eeb88a01fb660d2f` |
| `~/.claude/skills/ncu-report-skill/README.md` | 6,825 | `87fbd0242f92f94a` |
| `~/.claude/skills/ncu-report-skill/SKILL.md` | 8,302 | `3d713ab9b331ee69` |
| `~/.claude/skills/ncu-report-skill/blackwell-cuda-programming.md` | 45,341 | `9d77cd6f96f6e31f` |
| `~/.claude/skills/ncu-report-skill/helpers/README.md` | 2,990 | `5e4638d65366f758` |
| `~/.claude/skills/ncu-report-skill/helpers/analyze_reports.py` | 3,920 | `a47f2bb0b3ae415a` |
| `~/.claude/skills/ncu-report-skill/helpers/extract_stall_hotspots.py` | 5,791 | `973ed39db8e26fb0` |
| `~/.claude/skills/ncu-report-skill/helpers/harness_template.cu` | 7,724 | `7574aad7437e180a` |
| `~/.claude/skills/ncu-report-skill/helpers/list_flashinfer_workloads.py` | 11,386 | `5d38de992c0f1d42` |
| `~/.claude/skills/ncu-report-skill/helpers/ncu_utils.py` | 12,754 | `916d819d7d2a2e17` |
| `~/.claude/skills/ncu-report-skill/helpers/plot_timeline.py` | 4,799 | `f64527f03285eea0` |
| `~/.claude/skills/ncu-report-skill/helpers/safetensors_loader.h` | 6,801 | `1935d681420449ce` |
| `~/.claude/skills/ncu-report-skill/reference/00-directory-layout.md` | 7,903 | `7a6a19a8f56915f1` |
| `~/.claude/skills/ncu-report-skill/reference/01-workflow.md` | 10,067 | `04c66b136dc8b896` |
| `~/.claude/skills/ncu-report-skill/reference/02-harness-guide.md` | 11,752 | `9ca0b8b160530e0d` |
| `~/.claude/skills/ncu-report-skill/reference/03-collection.md` | 7,661 | `df62c9741e100327` |
| `~/.claude/skills/ncu-report-skill/reference/04-python-api.md` | 9,396 | `7f3e893060f87eef` |
| `~/.claude/skills/ncu-report-skill/reference/05-analysis-dimensions.md` | 15,769 | `4fe129838e54b456` |
| `~/.claude/skills/ncu-report-skill/reference/06-diagnosis-playbook.md` | 15,061 | `0f491f4e6937a16c` |
| `~/.claude/skills/ncu-report-skill/reference/07-report-template.md` | 6,396 | `c794959b8a85cd08` |
| `~/.claude/skills/ncu-report-skill/reference/08-b200-metric-names.md` | 11,518 | `6e9a5c5b45691956` |
| `~/.claude/skills/ncu-report-skill/reference/09-common-issues.md` | 8,398 | `63463464dfc34eb3` |

Synced bundle SKILL.md files (path, sha256[:16]); the remaining files are scripts/schemas/fonts listed in the JSON:

| path | bytes | sha256[:16] |
|---|---|---|
| `~/.claude/skills/synced/7d0b14af-b5b4-4698-bb94-0ecfa0c9d5db_0e2d18ca-a226-4c42-b4c1-2428ae9687a9/docs/SKILL.md` | 1,249 | `98e4fb5934042152` |
| `~/.claude/skills/synced/7d0b14af-b5b4-4698-bb94-0ecfa0c9d5db_0e2d18ca-a226-4c42-b4c1-2428ae9687a9/docx/SKILL.md` | 7,639 | `912cb15683a476a2` |
| `~/.claude/skills/synced/7d0b14af-b5b4-4698-bb94-0ecfa0c9d5db_0e2d18ca-a226-4c42-b4c1-2428ae9687a9/google-workspace/SKILL.md` | 13,869 | `f53b2d8c9bcb4336` |
| `~/.claude/skills/synced/7d0b14af-b5b4-4698-bb94-0ecfa0c9d5db_0e2d18ca-a226-4c42-b4c1-2428ae9687a9/import-memory/SKILL.md` | 10,689 | `5179b0780b7025c9` |
| `~/.claude/skills/synced/7d0b14af-b5b4-4698-bb94-0ecfa0c9d5db_0e2d18ca-a226-4c42-b4c1-2428ae9687a9/morning/SKILL.md` | 18,163 | `a7c4dbcb74579973` |
| `~/.claude/skills/synced/7d0b14af-b5b4-4698-bb94-0ecfa0c9d5db_0e2d18ca-a226-4c42-b4c1-2428ae9687a9/pdf/SKILL.md` | 8,072 | `9f78b8359fbd4943` |
| `~/.claude/skills/synced/7d0b14af-b5b4-4698-bb94-0ecfa0c9d5db_0e2d18ca-a226-4c42-b4c1-2428ae9687a9/pptx/SKILL.md` | 28,618 | `d8bf85f2ef187ce7` |
| `~/.claude/skills/synced/7d0b14af-b5b4-4698-bb94-0ecfa0c9d5db_0e2d18ca-a226-4c42-b4c1-2428ae9687a9/skill-creator/SKILL.md` | 33,351 | `42765c880e446a83` |
| `~/.claude/skills/synced/7d0b14af-b5b4-4698-bb94-0ecfa0c9d5db_0e2d18ca-a226-4c42-b4c1-2428ae9687a9/xlsx/SKILL.md` | 8,598 | `6712b39718fe8150` |

## 4. Legacy LLM docs (`llm_wt/tilebench/llm/`)

| path | bytes | lines | sha256[:16] | mtime | frontmatter name | notes |
|---|---|---|---|---|---|---|
| `llm_wt/tilebench/llm/system_prompt.md` | 890 | 15 | `9029eed239af68c1` | 2026-10-05T03:16:50 | none | read into `prompt_builder.SYSTEM` at import; no headings; forbids autotune |
| `llm_wt/tilebench/llm/framework_guide.md` | 15,075 | 346 | `223fc834c9c8bfbd` | 2026-10-05T03:16:50 | none | injected into every user message by both prompt builders; 13 `##` sections |

`prompt_builder._SKILL_PATHS = {'triton': REPO_ROOT/skills/triton-guide/SKILL.md, 'cutile': REPO_ROOT/skills/cutile-guide/SKILL.md}`; `_read_skill()` strips the YAML block, `lstrip()`s, and returns `''` when the file is missing. There is no tilelang or nki entry and no hash check.

## 5. Copy comparison

| pair | result |
|---|---|
| `skills/triton-guide` vs `.claude/skills/triton-guide` | byte-identical, sha256 `ac48c68a4d980979…`, 1644 lines |
| `skills/cutile-guide` (1140 lines, `7f25fdf978bd…`) vs `.claude/skills/cutile-guide` (1158 lines, `767f467757f8…`) | `.claude` adds 19 lines / removes 1: inserts `### 18.3 Compile-Time Loop Unrolling — ct.static_iter` (code example + prose on `ct.static_iter`/`static_eval`/`static_assert`, citing top_k_selection) and renumbers `### 18.3 Runtime Conditionals` to `### 18.4`. No other hunks. |
| `tilelang-guide`, `nki-guide` | single copies; nothing in repo or project `.claude` |

## 6. Versions: claimed by document vs installed (`tilebench_env`, Python 3.10.19)

| package | installed | claimed by doc | match |
|---|---|---|---|
| triton | 3.6.0 | triton-guide: **3.6.0** (explicit) | yes |
| cuda-tile | 1.5.0 | cutile-guide (both copies): **unknown** — no version string; API surface is `cuda.tile_experimental.autotune_launch` -> `TunedResult` (deprecated), no `ct.tune.exhaustive_search`/`TuningResult`/`with_hints` | no (stale API) |
| cuda-tile-experimental | 0.0.1 (present, from git) | cutile-guide imports `cuda.tile_experimental` guarded | n/a |
| tilelang | 0.1.11 | tilelang-guide: **0.1.11** (explicit, 'pinned') | yes |
| apache-tvm-ffi | 0.1.11 | tilelang-guide: **0.1.11** required | yes |
| torch | 2.10.0+cu130 | none claims a torch version | n/a |
| neuronxcc / nki / neuronx-cc | **ABSENT** | nki-guide: neuronxcc.nki (Neuron SDK 2.18–2.26) -> nki 0.1.0 (SDK 2.27), 0.3.0 GA (SDK 2.29), 0.5.0 (SDK 2.31) | cannot verify |
| cuda-bindings | 13.0.3 | — | n/a |
| tileiras / cuda-python / triton-proton | ABSENT | — | n/a |

`pip show tilelang` location: `…/tilebench_env/lib/python3.10/site-packages/tilelang` (subdirs `language/`, `autotuner/`, `jit/`, `src/`, `3rdparty/`; the wheel ships no `docs/`, only `backend/README.md` and `carver/README.md`). Paths noted only; nothing copied.

## 7. License

- Repo root: **no LICENSE file** (not tracked, not on disk); `README.md` §Attribution only states that third-party libraries keep their own licenses. Repo license = **unspecified**. The two public skills therefore inherit an unspecified license.
- `.claude/` and `~/.claude/` copies: local, unknown license, same author likely.
- `nki-guide`: PRIVATE until confirmed; body not exported here.
- `ncu-report-skill`: third-party clone (mit-han-lab), README says MIT.
- `synced/`: Anthropic-provided skill bundle, not project content.

## 8. Gaps

- No TileLang reference in the public repo: the only TileLang doc is ~/.claude/skills/tilelang-guide/SKILL.md (user-local, 13,880 B, pinned 0.1.11 = installed).
- NKI reference is private and user-local only (~/.claude/skills/nki-guide/SKILL.md); neuronxcc/nki are not installed in tilebench_env, so its claims cannot be verified here. Body must not be exported until confirmed.
- Both cuTile guides carry no version string and document the deprecated cuda.tile_experimental.autotune_launch/TunedResult path; neither mentions ct.tune.exhaustive_search, TuningResult, CutileAutotuner, kernel_with_hints or with_hints. Verified in llm_wt: all 45 tilebench/benchmarks/operators/*/impl_cutile.py import CutileAutotuner and use with_hints, 0 use autotune_launch; framework_guide.md's cuTile template also uses with_hints. The guides predate the installed cuda-tile 1.5.0.
- The two cuTile copies differ (.claude adds §18.3 ct.static_iter); the public repo copy is the older one. .claude/skills is git-excluded (.git/info/exclude), so the newer text is not reproducible from any branch.
- No device-context document exists anywhere: no GPU/accelerator spec doc in llm_wt (no docs/ device file; data/peak_performance/ named in CLAUDE.md does not exist in llm_wt or in the Tilebench checkout; the only in-code hardware facts are the LLC-calibration comments in tilebench/hardware.py), none in .claude/skills, none in ~/.claude/skills. Hardware facts are scattered inline: framework_guide.md 'Hardware constraints' (B200-only), triton-guide §20.2 (stale: 8 TB/s HBM, ~50 MB L2 vs the repo's measured 6539.4 GB/s and 126.5 MB), calculation-guide §4.4 (datasheet ridge-point table), nki-guide §2 (private). The only standalone Blackwell document on disk is third-party (ncu-report-skill/blackwell-cuda-programming.md, MIT, Chinese, CUDA C++ oriented, CUDA 13.2).
- No GH200, MI300X or Trainium device documents exist, although TileBench++ multiarch targets them.
- No skill file carries a 'version' frontmatter field; calculation-guide has no frontmatter at all. Manifests must pin sha256, not frontmatter.
- triton-guide §23.1 uses the pre-package path benchmarks/operators/<op>/ (now tilebench/benchmarks/operators/<op>/), §23.2 lists the pre-#313 flat results/ layout (results/csv/<op>_summary.csv), and §23.3 describes one run doing an autotune pass followed by a _DEFAULT_CONFIG timed pass, whereas the current engine runs a single mode per invocation (run_bench.py --autotune -> one `autotune` bool in tilebench/core/engine.py); §28 points to lib/Triton/triton-3.6.0/, which is not present in llm_wt. Content needs revision before injection.
- Repo root has no LICENSE file, so the shareability of the two public skills is formally unspecified.
- The legacy loader pins nothing (path only, frontmatter stripped at runtime); byte-identity of rebuilt iter-0 prompts depends on the skills/ bytes staying exactly as in commit 06320a62.

## 9. Section classification — `skills/triton-guide/SKILL.md` (Triton 3.6.0)

| line | level | heading | class |
|---|---|---|---|
| 9 | # | Triton Comprehensive Programming Guide | API/semantics |
| 17 | ## | 1. Import Boilerplate | API/semantics |
| 50 | ## | 2. Data Types | API/semantics |
| 52 | ### | 2.1 DType Constants (all on `tl`) | API/semantics |
| 70 | ### | 2.2 Dtype of a Pointer / Tensor | API/semantics |
| 79 | ### | 2.3 Common Casts | API/semantics |
| 94 | ## | 3. The Kernel Programming Model | API/semantics |
| 96 | ### | 3.1 Single Program = Single Tile | API/semantics |
| 106 | ### | 3.2 Block Tensors | API/semantics |
| 115 | ### | 3.3 Program ID & Num Programs | API/semantics |
| 134 | ### | 3.4 Grid Launch Syntax | API/semantics |
| 153 | ### | 3.5 Kernel Parameter Rules | API/semantics |
| 169 | ### | 3.6 tl.assume — Integer Bound Hints | API/semantics |
| 180 | ## | 4. Pointer & Offset Arithmetic | API/semantics |
| 182 | ### | 4.1 The Pointer-Arithmetic Model | API/semantics |
| 201 | ### | 4.2 Broadcasting (NumPy-style) | API/semantics |
| 207 | ### | 4.3 Masking OOB Accesses | API/semantics |
| 223 | ### | 4.4 Block Pointers (`make_block_ptr`) — Structured Pointer Arithmetic | API/semantics |
| 247 | ### | 4.5 Tensor Descriptors (TMA, Hopper / Blackwell) | API/semantics |
| 279 | ## | 5. Memory Operations | API/semantics |
| 281 | ### | 5.1 tl.load() | API/semantics |
| 314 | ### | 5.2 tl.store() | API/semantics |
| 332 | ### | 5.3 tl.gather() / tl.histogram() | API/semantics |
| 341 | ### | 5.4 Atomic Operations | API/semantics |
| 372 | ## | 6. Tile Creation | API/semantics |
| 394 | ## | 7. Shape & View Operations | API/semantics |
| 416 | ## | 8. Arithmetic & Math | API/semantics |
| 418 | ### | 8.1 Operator Overloads | API/semantics |
| 439 | ### | 8.2 Explicit Arithmetic | API/semantics |
| 453 | ### | 8.3 Math (tl.math.* — prefer these over libdevice) | API/semantics |
| 482 | ## | 9. Matrix Multiply — tl.dot() | API/semantics |
| 520 | ### | 9.1 tl.dot_scaled() — Microscaling FP8/FP4 | API/semantics |
| 536 | ## | 10. Reductions | API/semantics |
| 556 | ### | 10.1 Custom Reductions with tl.reduce() | API/semantics |
| 576 | ### | 10.2 Scans (Prefix) | API/semantics |
| 586 | ## | 11. Control Flow | API/semantics |
| 588 | ### | 11.1 Python for / if — Compile-time Specialization | API/semantics |
| 600 | ### | 11.2 tl.range — Runtime Loops with Attributes | API/semantics |
| 617 | ### | 11.3 tl.static_range — Compile-time for Loop | API/semantics |
| 626 | ### | 11.4 While Loops + tl.condition | API/semantics |
| 637 | ## | 12. Debug & Introspection | API/semantics |
| 651 | ## | 13. Compiler Hints | API/semantics |
| 674 | ## | 14. Randomness (Philox PRNG) | API/semantics |
| 694 | ## | 15. @triton.jit — The JIT Decorator | API/semantics |
| 726 | ## | 16. @triton.autotune — Configuration Search | API/semantics |
| 754 | ### | 16.1 triton.Config | API/semantics |
| 776 | ### | 16.2 best_config / Cached Result | benchmark/autotune workflow |
| 788 | ### | 16.3 Disk Cache | benchmark/autotune workflow |
| 792 | ### | 16.4 Verbose Logging | benchmark/autotune workflow |
| 802 | ## | 17. @triton.heuristics — Derived Meta-params | API/semantics |
| 819 | ## | 18. TileBench Integration — impl_triton.py Contract | benchmark/autotune workflow |
| 833 | ### | 18.1 Canonical Template | benchmark/autotune workflow |
| 896 | ### | 18.2 Why the Dual-Kernel Pattern | benchmark/autotune workflow |
| 903 | ### | 18.3 What NOT to do | benchmark/autotune workflow |
| 917 | ### | 18.4 Common `key=` Choices | benchmark/autotune workflow |
| 931 | ## | 19. Common Kernel Patterns | example kernel |
| 933 | ### | Pattern A: Element-wise (mul2, relu, vector_add, swiglu, dropout) | example kernel |
| 953 | ### | Pattern B: Row-wise single-pass (softmax with row-in-one-tile) | example kernel |
| 977 | ### | Pattern C: Row-wise two-pass (rmsnorm, layernorm, l2_norm) | example kernel |
| 1008 | ### | Pattern D: Matmul (BLOCK_M × BLOCK_K × BLOCK_N tiles with GROUP_SIZE_M grouping) | example kernel |
| 1058 | ### | Pattern E: Flash Attention (online softmax + streaming K/V) | example kernel |
| 1135 | ### | Pattern F: Persistent Kernel (one program handles many output tiles) | example kernel |
| 1153 | ### | Pattern G: Spin-Locked Parallel Reduction (layernorm backward dw/db) | example kernel |
| 1175 | ### | Pattern H: Using `tl.make_tensor_descriptor` (TMA, Hopper+) | example kernel |
| 1198 | ## | 20. Performance Methodology | tuning recommendation |
| 1200 | ### | 20.1 Occupancy / Resource Tradeoffs | tuning recommendation |
| 1209 | ### | 20.2 Saturating HBM Bandwidth | hardware spec |
| 1215 | ### | 20.3 Compute-Bound Checks | tuning recommendation |
| 1221 | ### | 20.4 triton.testing.do_bench | benchmark/autotune workflow |
| 1235 | ### | 20.5 Common `perf_report` Pattern (from tutorials) | benchmark/autotune workflow |
| 1254 | ## | 21. Debugging Checklist | other |
| 1267 | ### | 21.1 Dumping IRs | other |
| 1274 | ### | 21.2 Interpreter Mode (pure Python, for debugging) | other |
| 1284 | ## | 22. Critical Constraints & Gotchas | API/semantics |
| 1286 | ### | 22.1 constexpr-ness of Shapes | API/semantics |
| 1290 | ### | 22.2 `tl.advance` Returns a New Pointer | API/semantics |
| 1302 | ### | 22.3 Arithmetic on Blocks Returns New Blocks (Immutable) | API/semantics |
| 1313 | ### | 22.4 Always Mask or Pad Non-Pow-2 Trailing Dims | API/semantics |
| 1323 | ### | 22.5 Compute in fp32, Store in Source Dtype | API/semantics |
| 1333 | ### | 22.6 bfloat16 Is Promoted Silently in Reductions | API/semantics |
| 1337 | ### | 22.7 `num_stages` on Loops vs on Kernels | API/semantics |
| 1344 | ### | 22.8 Host-side `triton.cdiv` vs In-Kernel `tl.cdiv` | API/semantics |
| 1356 | ### | 22.9 `triton.next_power_of_2` Is Host-Only | API/semantics |
| 1365 | ### | 22.10 Tensor Pointers vs Block Pointers — Don't Mix | API/semantics |
| 1372 | ### | 22.11 Float8 GEMM — `ct.mma`-Style Restrictions in Triton | API/semantics |
| 1378 | ### | 22.12 Kernel Must Not Mutate Input Tensors Unless Documented | benchmark/autotune workflow |
| 1384 | ## | 23. TileBench-Specific Workflow | benchmark/autotune workflow |
| 1386 | ### | 23.1 File Skeleton | benchmark/autotune workflow |
| 1398 | ### | 23.2 Running & Profiling | benchmark/autotune workflow |
| 1417 | ### | 23.3 Engine Behavior | benchmark/autotune workflow |
| 1428 | ### | 23.4 config.yaml Example | benchmark/autotune workflow |
| 1456 | ## | 24. Quick Reference Tables | API/semantics |
| 1458 | ### | 24.1 tl.load() Parameters | API/semantics |
| 1471 | ### | 24.2 tl.store() Parameters | API/semantics |
| 1482 | ### | 24.3 tl.dot() Dtype Matrix | API/semantics |
| 1494 | ### | 24.4 `triton.Config` Fields | API/semantics |
| 1506 | ### | 24.5 @triton.autotune Args | API/semantics |
| 1518 | ### | 24.6 `tl.range` Attributes | API/semantics |
| 1531 | ## | 25. Cheatsheet — Minimal Kernels | example kernel |
| 1533 | ### | Vector Add | example kernel |
| 1546 | ### | Fused Softmax (single-pass) | example kernel |
| 1559 | ### | RMSNorm (two-pass) | example kernel |
| 1583 | ### | Matmul with Grouped Scheduling | tuning recommendation |
| 1587 | ### | Flash Attention Forward | tuning recommendation |
| 1593 | ## | 26. When cuTile and Triton Disagree | other |
| 1612 | ## | 27. Environment Variables (Most Useful) | other |
| 1627 | ## | 28. Further Reading | other |

Notes: §4.5 (TMA descriptors) and §22.11 (fp8 on Hopper/Blackwell) are API sections that embed hardware constraints; §20.2 is classified hardware spec because its only content is B200 numbers (stale: 8 TB/s, ~50 MB L2); §18.4 and §23.4 contain TileBench-contract material, not DSL API.

## 10. Section classification — `skills/cutile-guide/SKILL.md` (repo copy; `.claude` copy identical except §18.3/18.4)

| line | level | heading | class |
|---|---|---|---|
| 9 | # | cuTile (cuda.tile) Comprehensive Programming Guide | API/semantics |
| 15 | ## | 1. Import Boilerplate | API/semantics |
| 43 | ## | 2. Data Types | API/semantics |
| 45 | ### | 2.1 DType Constants | API/semantics |
| 59 | ### | 2.2 DType Usage in Kernels | API/semantics |
| 71 | ## | 3. Core Classes | API/semantics |
| 73 | ### | 3.1 Array (Global Memory) | API/semantics |
| 93 | ### | 3.2 Tile (Local Data) | API/semantics |
| 122 | ### | 3.3 Scalar | API/semantics |
| 128 | ## | 4. Enumerations | API/semantics |
| 130 | ### | 4.1 PaddingMode | API/semantics |
| 143 | ### | 4.2 RoundingMode | API/semantics |
| 156 | ### | 4.3 MemoryOrder (for atomics) | API/semantics |
| 165 | ### | 4.4 MemoryScope (for atomics) | API/semantics |
| 175 | ## | 5. Kernel Definition & Launch | API/semantics |
| 177 | ### | 5.1 @ct.kernel Decorator | API/semantics |
| 207 | ### | 5.2 @ct.function Decorator | API/semantics |
| 221 | ### | 5.3 ct.launch() | API/semantics |
| 232 | ### | 5.4 Block/Grid Information | API/semantics |
| 246 | ## | 6. Memory Operations | API/semantics |
| 248 | ### | 6.1 ct.load() | API/semantics |
| 277 | ### | 6.2 ct.store() | API/semantics |
| 299 | ### | 6.3 ct.gather() | API/semantics |
| 328 | ### | 6.4 ct.scatter() | API/semantics |
| 352 | ## | 7. Atomic Operations | API/semantics |
| 385 | ## | 8. Tile Creation (Factory Functions) | API/semantics |
| 409 | ## | 9. Shape & View Operations | API/semantics |
| 436 | ## | 10. Type Conversion | API/semantics |
| 456 | ## | 11. Arithmetic Operations | API/semantics |
| 458 | ### | 11.1 Operator Overloads (preferred for simple cases) | API/semantics |
| 470 | ### | 11.2 Explicit Functions (when you need extra control) | API/semantics |
| 496 | ### | 11.3 Bitwise Operations | API/semantics |
| 509 | ## | 12. Comparison Operations | API/semantics |
| 511 | ### | 12.1 Operator Overloads (return bool tiles) | API/semantics |
| 518 | ### | 12.2 Explicit Functions | API/semantics |
| 531 | ## | 13. Math Functions | API/semantics |
| 533 | ### | 13.1 Exponential & Logarithmic | API/semantics |
| 544 | ### | 13.2 Trigonometric | API/semantics |
| 551 | ### | 13.3 Power & Root | API/semantics |
| 559 | ### | 13.4 Rounding | API/semantics |
| 568 | ## | 14. Reduction Operations | API/semantics |
| 593 | ### | 14.1 Custom Reduction | API/semantics |
| 620 | ## | 15. Scan (Prefix) Operations | API/semantics |
| 639 | ## | 16. Conditional & Selection | API/semantics |
| 662 | ## | 17. Matrix Multiply | API/semantics |
| 664 | ### | 17.1 ct.mma() — Matrix Multiply-Accumulate (Fused) | API/semantics |
| 696 | ### | 17.2 ct.matmul() — Simple Matrix Multiply | API/semantics |
| 708 | ## | 18. Control Flow | API/semantics |
| 710 | ### | 18.1 Python for-loop (compile-time unrolled) | API/semantics |
| 721 | ### | 18.2 Python if/else (compile-time) | API/semantics |
| 728 | ### | 18.3 Runtime Conditionals | API/semantics |
| 736 | ## | 19. Synchronization (Advanced) | API/semantics |
| 759 | ## | 20. TensorView & PartitionView (Advanced) | API/semantics |
| 786 | ## | 21. Debug & Utility | API/semantics |
| 788 | ### | 21.1 printf | API/semantics |
| 803 | ### | 21.2 assert_ | API/semantics |
| 813 | ## | 22. Compiler Hints (Advanced) | API/semantics |
| 834 | ## | 23. Autotune Pattern | benchmark/autotune workflow |
| 836 | ### | 23.1 autotune_launch() | API/semantics |
| 861 | ### | 23.2 clear_autotune_cache() | API/semantics |
| 869 | ### | 23.3 Standard Implementation Template | benchmark/autotune workflow |
| 911 | ## | 24. run() and get_last_config() Contract | benchmark/autotune workflow |
| 931 | ## | 25. Common Kernel Patterns | example kernel |
| 933 | ### | Pattern A: Element-wise (mul2, relu, vector_add, dropout, swiglu) | example kernel |
| 951 | ### | Pattern B: Row-wise single-pass (softmax, argmax) | example kernel |
| 969 | ### | Pattern C: Row-wise two-pass (rmsnorm, layernorm, l2_norm) | example kernel |
| 996 | ### | Pattern D: Flash Attention (online softmax + MMA) | example kernel |
| 1026 | ### | Pattern E: Dynamic indexing (gather/scatter) | example kernel |
| 1038 | ### | Pattern F: NotImplementedError | example kernel |
| 1050 | ## | 26. Critical Constraints & Gotchas | API/semantics |
| 1052 | ### | Tile dimensions must be powers of two | API/semantics |
| 1058 | ### | ct.store() requires static indices | API/semantics |
| 1061 | ### | Tiles are immutable | API/semantics |
| 1068 | ### | Always accumulate in fp32 | API/semantics |
| 1075 | ### | Grid tuple | API/semantics |
| 1082 | ### | flush_to_zero for exp2 | API/semantics |
| 1088 | ### | Kernel vs host code | API/semantics |
| 1093 | ### | Autotune has no persistent cache | benchmark/autotune workflow |
| 1096 | ### | fp16 argmax precision | API/semantics |
| 1104 | ## | 27. Quick Reference Tables | API/semantics |
| 1106 | ### | ct.load() Parameters | API/semantics |
| 1118 | ### | ct.store() Parameters | API/semantics |
| 1129 | ### | MMA Supported Dtypes | API/semantics |

Notes: §5.1 embeds `ByTarget(sm_100, sm_120)` and `num_ctas`/`occupancy` hint ranges (API); §23.1–23.2 document the deprecated `autotune_launch`/`clear_autotune_cache` API; §23.3 and §24 are TileBench contract. The `.claude` copy inserts `### 18.3 Compile-Time Loop Unrolling — ct.static_iter` (API/semantics) at line 728 and shifts Runtime Conditionals to line 746.

## 11. Section classification — `~/.claude/skills/tilelang-guide/SKILL.md` (TileLang 0.1.11)

| line | level | heading | class |
|---|---|---|---|
| 6 | # | TileLang — Authoring & Review Guide (pinned to v0.1.11) | API/semantics |
| 14 | ## | 1. Two authoring styles: lazy vs eager `@tilelang.jit` | API/semantics |
| 36 | ## | 2. `T.const` vs `T.dynamic` vs `T.symbolic` — recompilation semantics | API/semantics |
| 52 | ## | 3. `T.copy` — extent inference and OOB (the subtle correctness trap) | API/semantics |
| 70 | ## | 4. `T.Kernel`, tensors, allocation | API/semantics |
| 78 | ## | 5. Compute constructs | API/semantics |
| 89 | ## | 6. Math, casting, control flow, atomics, RNG | API/semantics |
| 100 | ## | 7. `@tilelang.autotune` + the TileBench config contract | benchmark/autotune workflow |
| 135 | ## | 8. Review red-flags checklist | other |

Notes: §2 and §3 are API semantics with embedded performance consequences (recompilation, zero-fill); §6 ends with target detection (`sm_100a`) and a Blackwell support statement; §7 is the TileBench `get_last_config` contract; §8 is a review checklist.

## 12. Section classification — `~/.claude/skills/nki-guide/SKILL.md` (headings only, PRIVATE)

| line | level | heading | class |
|---|---|---|---|
| 6 | # | AWS Neuron NKI — Authoring & Review Guide | API/semantics |
| 14 | ## | 1. Imports & kernel definition | API/semantics |
| 36 | ## | 2. The hardware model (the source of every constraint) | hardware spec |
| 49 | ### | The load-bearing tile rules | hardware spec |
| 58 | ## | 3. Memory ops, buffers, casting | API/semantics |
| 70 | ## | 4. `mask` — a compile-time predicate, not a runtime tensor | API/semantics |
| 91 | ## | 5. Elementwise, reductions, matmul, transpose | API/semantics |
| 111 | ## | 6. `nisa.*` low-level ISA — destination-passing convention | API/semantics |
| 128 | ## | 7. Loops & SPMD | API/semantics |
| 136 | ## | 8. Timing / testing (and the TileBench NKI backend) | benchmark/autotune workflow |
| 145 | ## | 9. Review red-flags checklist | other |

159 lines, 13,535 B; frontmatter `name: nki-guide` + description. Classification is by heading title only; body intentionally not read into this report.

## 13. Section classification — `tilebench/llm/framework_guide.md` and `.claude/skills/calculation-guide/SKILL.md`

framework_guide.md:

| line | level | heading | class |
|---|---|---|---|
| 1 | # | TileBench Framework Guide | benchmark/autotune workflow |
| 6 | ## | Files you must produce | benchmark/autotune workflow |
| 17 | ## | Required exports (both impls) | benchmark/autotune workflow |
| 32 | ## | ⛔️ No autotune — you pick one configuration per iteration | benchmark/autotune workflow |
| 50 | ## | Triton template (no autotune) | example kernel |
| 108 | ## | cuTile template (no autotune) | example kernel |
| 164 | ## | Choosing tile / BLOCK sizes | tuning recommendation |
| 204 | ## | Multi-dtype support | tuning recommendation |
| 229 | ## | Hardware constraints to remember (B200, sm_100) | hardware spec |
| 242 | ## | ⛔️ FORBIDDEN: delegating the actual computation to PyTorch / cuDNN / cuBLAS | benchmark/autotune workflow |
| 279 | ## | ⛔️ FORBIDDEN: caching outputs across `run()` calls | benchmark/autotune workflow |
| 294 | ## | Common pitfalls (the harness rejects these) | benchmark/autotune workflow |
| 309 | ## | What the engine does with your code | benchmark/autotune workflow |
| 336 | ## | TL;DR checklist before you return code | benchmark/autotune workflow |

calculation-guide (not a DSL reference; listed for completeness):

| line | level | heading | class |
|---|---|---|---|
| 1 | # | SKILLS.md — LLM Inference Profiling Annotations for GPT-OSS | other |
| 7 | ## | 1. Overview | other |
| 31 | ## | 2. GPT-OSS Model Architecture Reference | other |
| 61 | ## | 3. Scope Annotations | other |
| 63 | ### | 3.1 What is a Scope Annotation? | other |
| 88 | ### | 3.2 Scope Hierarchy for GPT-OSS | other |
| 113 | ### | 3.3 How to Annotate: Rules | other |
| 131 | ### | 3.4 Existing Annotations in GPT-OSS | other |
| 143 | ## | 4. Metric Calculations via Triton `launch_metadata` | API/semantics |
| 145 | ### | 4.1 Core Mechanism | API/semantics |
| 149 | ### | 4.2 Canonical Template | API/semantics |
| 215 | ### | 4.3 How to Determine Compute-bound vs. Memory-bound | API/semantics |
| 251 | ### | 4.4 GPU Ridge Point Reference Table | hardware spec |
| 260 | ### | 4.5 FLOPs and Bytes Formulas per Operation | other |
| 340 | ### | 4.6 Writing a `metadata_fn`: Step-by-Step | API/semantics |
| 403 | ## | 5. Phase-Aware Profiling: Prefill vs. Decode | other |
| 428 | ## | 6. Putting It Together: Two Paths | other |
| 430 | ### | Path 1: Triton kernels — use `launch_metadata` | other |
| 464 | ### | Path 2: Non-Triton ops — use `proton.scope` with metrics | other |
| 481 | ## | 7. Checklist for the Agent | other |
| 499 | ## | 8. Reference: Existing Triton Kernel Metadata in triton_kernels | other |
| 515 | ## | 9. File Locations in GPT-OSS | other |
