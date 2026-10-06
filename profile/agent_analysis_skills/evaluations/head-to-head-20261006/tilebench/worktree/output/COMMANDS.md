# Saved-Evidence Commands

All commands ran in this trial's `tilebench/worktree`. No benchmark, autotune,
GPU workload, installation, network access, Git history, commit or push was used.
Only `output/` was written. The shell's pre-existing Lmod/posix warning did not
prevent any analysis command from completing.

Read scope and local skill references:

```bash
pwd
sed -n '1,240p' .agents/skills/tilebench-comparison/SKILL.md
sed -n '1,280p' .agents/skills/tilebench-comparison/references/comparison.md
sed -n '1,280p' .agents/skills/tilebench-analysis/references/tilebench-layout.md
sed -n '1,280p' .agents/skills/tilebench-analysis/references/diagnosis.md
sed -n '1,280p' .agents/skills/tilebench-analysis/references/nvidia-ncu.md
sed -n '1,280p' .agents/skills/tilebench-analysis/references/b200.md
rg --files .agents evidence tools tilebench/benchmarks/operators/2d_max_pooling
sed -n '1,310p' tilebench/benchmarks/operators/2d_max_pooling/impl_tilelang.py
sed -n '1,310p' tilebench/benchmarks/operators/2d_max_pooling/impl_triton.py
sed -n '1,310p' tilebench/benchmarks/operators/2d_max_pooling/impl_cutile.py
sed -n '1,310p' tilebench/benchmarks/operators/2d_max_pooling/config.yaml
sed -n '1,280p' evidence/reports.json
```

Reproduce the final evidence exports with the only approved Python interpreter:

```bash
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python /scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py evidence/tilelang_fp32.ncu-rep --output output/tilelang_metrics.json
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python /scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py evidence/triton_fp32.ncu-rep --output output/triton_metrics.json
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python /scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py evidence/cutile_fp32.ncu-rep --output output/cutile_metrics.json
/opt/nvidia/nsight-compute/2026.1.1/ncu --import evidence/tilelang_fp32.ncu-rep --page source --print-source sass --log-file output/tilelang_sass.txt
/opt/nvidia/nsight-compute/2026.1.1/ncu --import evidence/triton_fp32.ncu-rep --page source --print-source sass --log-file output/triton_sass.txt
/opt/nvidia/nsight-compute/2026.1.1/ncu --import evidence/cutile_fp32.ncu-rep --page source --print-source sass --log-file output/cutile_sass.txt
PYTHONPATH=.:/opt/nvidia/nsight-compute/2026.1.1/extras/python /scratch/arustagi/tilebench_pdf_env/bin/python output/extract_case.py
/scratch/arustagi/tilebench_pdf_env/bin/python output/build_evidence.py
```

Independent CSV check, using named numerator/denominator columns:

```bash
/scratch/arustagi/tilebench_pdf_env/bin/python tools/csv_ratios.py results/B200/csv/2d_max_pooling_autotune.csv --identity '{"params":"H=640","dtype":"fp32"}' --target-column tilelang_ms --reference-column triton_ms --reference-column cutile_ms
```

Exploratory Python one-liners inventoried `dir(report)`, `dir(action)`,
`dict(action.source_files())`, metric names/values, and opcode correlation IDs.
Their durable equivalents are in `extract_case.py`. Additional reads inspected
the winner JSON tails, tool source, SASS lines, and embedded/current diffs.
There were no failed NCU imports or missing requested scalar extractions.
