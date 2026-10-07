# Offline Commands

All commands ran with the working directory explicitly set to this isolated worktree:

```
/home/arustagi/repos/Tilebench-profile-study/profile/agent_skill_eval/20261006_head_to_head_batch2/gaussian_blur/original_ncu/worktree
```

Read `.agents/skills/ncu-report-skill/SKILL.md` and references 00, 04, 05, 06, and 07.
Read the three operator implementation files, `config.yaml`, the specified CSV, winner logs, `evidence/reports.json`, and local extraction helpers.
No parent workspaces, other trials, external reports, history, or human analyses were read.

Initial full scalar extraction, once per backend (replace BACKEND with tilelang, triton, cutile):

```sh
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python \
/scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py \
  evidence/BACKEND_fp32.ncu-rep --output output/metrics_all_BACKEND.json
```

Reproduce all selected scalar/per-PC records, hashes, embedded source, SASS, rule results, and winner selection:

```sh
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python \
/scratch/arustagi/tilebench_pdf_env/bin/python output/analyze_saved.py
```

Source correspondence checks (exit 1 means a textual difference was found):

```sh
diff -u output/embedded_triton_0.py tilebench/benchmarks/operators/gaussian_blur/impl_triton.py
diff -u output/embedded_cutile_0.py tilebench/benchmarks/operators/gaussian_blur/impl_cutile.py
```

Also used read-only `rg`, `sed`, `ls`, and pinned-Python one-liners to inspect extracted counters and report API methods.
The extractor enumerated complete range/action inventories and available metric names, rather than assuming the first action without checking.
Per-PC records retain exact instance indexes and correlation IDs. Units are the API's units, including empty strings where appropriate.
The saved-evidence instruction overrides skill harness, full/source capture, lineinfo rebuild, and GPU/process-check steps.
Only files under `output/` were created; inputs were unchanged. No GPU commands, benchmark execution, autotuning, network calls, installations, commits, or pushes occurred.
