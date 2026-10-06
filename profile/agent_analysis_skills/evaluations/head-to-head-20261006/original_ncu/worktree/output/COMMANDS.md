# Saved-evidence analysis commands

All commands run from this isolated worktree. Only the supplied PDF-era Python executable was invoked; NCU decoding is CPU-only. No benchmark, CUDA compilation, profiling collection, GPU access, package installation, network, commit, or push was run.

```sh
cd /home/arustagi/repos/Tilebench-profile-study/profile/agent_skill_eval/20261006_head_to_head_maxpool/original_ncu/worktree
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python /scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py evidence/tilelang_fp32.ncu-rep --output output/metrics_all_tilelang.json
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python /scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py evidence/triton_fp32.ncu-rep --output output/metrics_all_triton.json
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python /scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py evidence/cutile_fp32.ncu-rep --output output/metrics_all_cutile.json
/scratch/arustagi/tilebench_pdf_env/bin/python output/inspect_reports.py
/scratch/arustagi/tilebench_pdf_env/bin/python output/select_evidence.py
```

Read-only inspection used `rg`, `nl -ba`, `sed`, and the same Python for JSON/CSV parsing and ratio arithmetic. The installed local skill and analysis/API/playbook/report references were read. Source inputs were the current pooling implementation files, configuration, supplied winner logs, reports manifest, and CSV; no other operator was analyzed.

`inspect_reports.py` collects SASS PCs from source-correlated metrics and fills intermediate instruction addresses, verifies them through `sass_by_pc(address)`, saves PTX mapping, NCU rules, imported source content and per-PC stalls. `select_evidence.py` checks all manifest hashes, selects 126 exact scalar records, saves complete action inventories, verifies embedded kernel-body AST correspondence where available, and recalculates the CSV ratios.

The skill snapshot's zero-argument `sass_by_pc()` example failed against this API; its actual method requires an address. A source-file API mapping also required conversion to `dict` before JSON serialization. Both were corrected in the saved script. Initial Bash invocations emitted an unrelated module-system startup warning; subsequent inspection used `/bin/sh`. No raw input was modified.
