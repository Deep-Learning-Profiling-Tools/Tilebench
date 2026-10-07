# Offline Commands

All commands producing analysis artifacts ran in this trial worktree. Only the supplied saved reports and current operator source were used for the diagnosis.
The original NCU skill and its directory, Python API, six-dimension, diagnosis, report-template, and B200 metric-name references were read; saved-evidence restrictions superseded collection/harness instructions.

```bash
cd /home/arustagi/repos/Tilebench-profile-study/profile/agent_skill_eval/20261006_head_to_head_batch2/batched_matmul/original_ncu/worktree
export PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python
/scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py evidence/tilelang_fp16.ncu-rep --output output/all_tilelang.json
/scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py evidence/triton_fp16.ncu-rep --output output/all_triton.json
/scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py evidence/cutile_fp16.ncu-rep --output output/all_cutile.json
/scratch/arustagi/tilebench_pdf_env/bin/python output/analyze_saved.py
diff -u output/embedded_triton_impl_triton.py.txt tilebench/benchmarks/operators/batched_matmul/impl_triton.py
diff -u output/embedded_cutile_impl_cutile.py.txt tilebench/benchmarks/operators/batched_matmul/impl_cutile.py
```

`diff` exits 1 because the whole files differ as documented. `analyze_saved.py` compares individual function ASTs, verifies manifest hashes, inventories every range/action, and retains selected scalar and nonzero per-PC stall records.
SASS/PTX extraction uses the exact `launch__function_pcs` instance as a base and walks saved instructions in 16-byte increments until no instruction is available, without compiling or executing a kernel.
Source inspection used `nl -ba`, `sed -n`, and scoped `rg -n` on the current operator and extracted report code.
Exploratory API calls established that `sass_by_pc` requires an absolute instruction address; missing counters were inventoried, not interpreted as zeros.

No profiling commands were executed. Original capture flags, original runtime arguments, and the complete capture harness are unavailable, so no invented capture command is presented as reproduction.

An initial independent file-list command accidentally inherited the parent working directory and listed file names before the explicit trial directory was reapplied. No parent source, report, PDF, or human-analysis contents were read or used; all subsequent commands set this worktree explicitly.
