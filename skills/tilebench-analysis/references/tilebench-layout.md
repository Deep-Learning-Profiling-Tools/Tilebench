# Repository and Artifact Map

Use this map to locate one case, not to read the whole repository. It describes
TileBench's package-based layout; verify paths in the assigned checkout.
Other branches may have a different layout. Do not switch branches or borrow
another worktree's data silently to make this map fit.

## Resolve Hardware Before Choosing Tools

1. Start with the requested hardware or supplied artifact namespace. For released
   profiles, map `NVIDIA_B200` -> local label `B200`, `NVIDIA_GH200` -> `GH200`,
   and `AMD_MI300X` -> `MI300X`. These select evidence, not a runtime device.
2. Confirm using matching provenance/capture metadata: device name, vendor,
   capability or GCN architecture, source version, and software stack. A path label
   is a candidate identity, not proof. Inspect the saved report's inventoried device
   attributes when available; do not assume every report collected those attributes.
3. If the requested hardware and artifact metadata conflict, report the mismatch
   before using the capture to explain that case. If several hardware namespaces
   fit and the task names none, ask which experiment to investigate.
4. Route NVIDIA `.ncu-rep` files to the NCU API and matching B200/GH200 reference.
   Route MI300X directories to ROCm artifact discovery, not NCU. AMD navigation is
   supported here; AMD diagnosis is not provided by this skill.

Do not run device detection to identify an old capture: the agent's host may have
no GPU or a different GPU. `nvidia-smi`/runtime device checks are relevant if new
collection is explicitly authorized, not a prerequisite for reading saved reports.

## Source and Results

| Evidence | Main-branch location | How to use it |
|---|---|---|
| Operator semantics, case grid, formulas | `tilebench/benchmarks/operators/<op>/config.yaml` | Resolve dtype/shape and useful work. |
| Implementation | Same directory, `impl_<backend>.py` | Inspect only the requested backend; `impl_torch.py` is the semantic reference when needed. |
| Benchmark summary | `results/<hardware>/csv/<op>_<mode>.csv` | Match the exact params/dtype row; inspect actual columns and valid numeric entries. |
| Timing and winning-config logs | `results/<hardware>/logs/{time_measurement_logs,autotune_logs}/<op>_<mode>_<backend-tag>.json` | Match the explicitly identified run, not the newest glob match. |
| Run provenance | `results/<hardware>/logs/provenance/<op>_<mode>_<backend-tag>.json` | Check source, software, device, timing mode, and named outputs. |
| Canonical path helpers | `tilebench/paths.py` | Confirm namespace and naming conventions without importing the benchmark engine. |
| Backend selection | `tilebench/backends.py` | Backend tags have canonical order `triton-cutile-tilelang-nki`; they identify the run selection, not hardware support. |
| Hardware-specific dispatch | `tilebench/hardware.py` | Actual detected device selects code paths; `--gpu` is a result label, not a device selector. |

CSV timing columns can exist without valid measurements. Reject empty, nonfinite,
failed, or unsupported entries rather than treating them as zero. Autotune logs,
source snapshots and generated code are not guaranteed to be included in a fresh
clone. Do not infer a winner from a default config or the fastest candidate you
happen to find. A current winner does not automatically describe an older capture.

NKI measurements may appear beside GPU results, but run on Trainium. Their
device-local baseline is `torch_nki_ms`, not the NVIDIA/AMD `torch_ms` column.
Directory proximity does not make those measurements a same-hardware comparison.

## Saved Profiles

NVIDIA local reports:
`outputs/ncu/<hardware>/<op>/<backend>_<dtype>.ncu-rep`.
Catalogue and kernel metadata:
`outputs/profiling/<hardware>/{ncu_catalogue.json,kernel_counts.json}`.
The catalogue includes maximum-input cases and recorded configs; kernel metadata
helps check action coverage. Check associated capture logs/manifests when present.
Do not use another hardware namespace as a fallback or regenerate metadata as an
analysis step. Partial captures and reduced metric sets require explicit limits.

Released artifacts are hosted at
[bcui2/NCU_report](https://huggingface.co/datasets/bcui2/NCU_report).
Its NVIDIA prefixes are `NVIDIA_B200` and `NVIDIA_GH200`, rather than the local
`B200`/`GH200` labels. Their report paths retain `<op>/<backend>_<dtype>.ncu-rep`.
Inspect the selected revision and available paths; coverage changes over time.
Use workspace credentials without printing them. A download must be within the
task's authorized scope and output location, not an automatic whole-dataset fetch.

Generated code may be embedded in a report or retained in capture-specific
harness/compiler-cache directories. These are not guaranteed main-branch assets.
Check the supplied capture's inventory before declaring source or SASS absent.
Prefer its snapshot to current source; label current-source inspection if the
capture-time version cannot be established.

AMD local artifacts use
`outputs/rocprof_compute/<hardware>/<op>/<backend>_<dtype>/`, with `workload/`,
`pc_sampling/`, `analysis/`, `logs/`, and `capture.json`. The released AMD namespace
is `AMD_MI300X`; these are not `.ncu-rep` files. The ROCm navigation/collection code
lives in `tilebench/profiling/rocprof_compute.py` and
`scripts/profiling/rocprof_compute_*.py`. This skill does not yet diagnose them.

## Checkout and Collection Boundaries

Main's CLI tools are under `scripts/profiling/`; libraries are under
`tilebench/profiling/`. Backend support differs by entrypoint: at the inspected
revision, `ncu_driver.py` supports TileLang selection but `ncu_one.py` offers only
Triton/cuTile. Inspect the selected tool's source/options before recommending it.
Never bypass the existing harness with an improvised benchmark during analysis.

Other checkouts may use `tilebench_ops/`, `tilebench_run/`,
`results/csv/`, and `profile/<op>/<run>/manifest.json` plus `harness/`/`reports/`.
Use the manifest and workspace instructions there. Do not embed private `/scratch`
paths, a particular Python executable, or that study's package pins into the
distributed skill. Record the actual experiment's stack instead.

## Tool Roles, Not Automatic Commands

| Repo tool | Role | Analysis boundary |
|---|---|---|
| `scripts/run_bench.py`, `run_bench_all.py` | Generate benchmark measurements/config logs | Do not run to navigate existing results. |
| `scripts/profiling/ncu_catalogue.py`, `probe_kernel_count.py` | Generate per-hardware case/config and kernel metadata | Read existing metadata first; these execute work, not just inventory files. |
| `scripts/profiling/ncu_driver.py`, `ncu_one.py` | Collect NVIDIA profiles through the repo harness | New collection requires authorization; inspect actual backend options. |
| `scripts/profiling/ncu_generic_harness.py` | Workload executed by NCU | Inspect for capture semantics, not as a standalone timing replacement. |
| `scripts/profiling/rocprof_compute_driver.py`, `rocprof_compute_harness.py` | Collect AMD counter/PC-sampling artifacts | Do not invoke in an NVIDIA-only environment or substitute NCU definitions. |
| `scripts/profiling/hf_upload.py`, `hf_upload_rocm_compute.py` | Publish captured artifacts | Upload tools, not download/navigation tools; do not run during analysis. |

For downloads use the released-artifact reference; for saved NCU extraction use
the bundled helper. Neither operation re-runs a kernel.
