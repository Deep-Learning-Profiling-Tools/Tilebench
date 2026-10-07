# Saved-Evidence Commands

All commands ran in this isolated worktree. No GPU work, benchmarks, autotuning,
network, installation, history reads, commits or pushes were performed. Input files
were read only. Shell used `/bin/sh` after initial Bash reads printed an Lmod error.
The interpreter was exclusively `/scratch/arustagi/tilebench_pdf_env/bin/python`.
The reader was `/opt/nvidia/nsight-compute/2026.1.1/extras/python/ncu_report.py`.

## Guidance and Inputs

Read `.agents/skills/tilebench-comparison/SKILL.md`, its `references/comparison.md`,
and sibling analysis references `tilebench-layout.md`, `nvidia-ncu.md`, `b200.md`,
and `diagnosis.md` with `sed`. Read the three operator implementations/config,
the two supplied winner logs, `evidence/reports.json`, and the CSV. Narrow `rg`
queries located M=640 FP16 winners, metric names, saved SASS transfer/tensor/wait
instructions, and current harness cache comments. No human analysis was consulted.

## Exact Extraction

These three independent saved-report reads were run concurrently:

```sh
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python /scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py evidence/tilelang_fp16.ncu-rep --output output/tilelang_ncu.json
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python /scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py evidence/triton_fp16.ncu-rep --output output/triton_ncu.json
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python /scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py evidence/cutile_fp16.ncu-rep --output output/cutile_ncu.json
```

Each export had no scalar-extraction errors and one action. Complete inventories
remain in those JSONs; selected quantitative evidence is in `selected_records.json`.
Per-instance values were not used for quantitative claims.

## Saved Machine Code

```sh
/opt/nvidia/nsight-compute/2026.1.1/ncu --import evidence/tilelang_fp16.ncu-rep --page source --print-source sass > output/tilelang_sass.txt
/opt/nvidia/nsight-compute/2026.1.1/ncu --import evidence/triton_fp16.ncu-rep --page source --print-source sass > output/triton_sass.txt
/opt/nvidia/nsight-compute/2026.1.1/ncu --import evidence/cutile_fp16.ncu-rep --page source --print-source sass > output/cutile_sass.txt
```

These were successful offline imports. Listing parser returned `ok` with no
warnings for all three. Static instruction sites were not treated as dynamic work.

## Reproduce Derived Outputs

`inspect_evidence.py` was added with `apply_patch`. It verifies SHA256s against
the supplied manifest; exports embedded source text/inventories; parses SASS;
checks Triton/cuTile kernel AST equality; uniquely selects the CSV row; confirms
the TileLang logs agree; and writes comparison/coverage/exact-selection artifacts.

```sh
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python /scratch/arustagi/tilebench_pdf_env/bin/python output/inspect_evidence.py
```

`REPORT.md` and this file were authored with `apply_patch`. Quantitative report
claims use report range 0/action 0 throughout. CSV ratios derive from the printed
latencies, not stored rounded speedup columns or winner-log timing values.
