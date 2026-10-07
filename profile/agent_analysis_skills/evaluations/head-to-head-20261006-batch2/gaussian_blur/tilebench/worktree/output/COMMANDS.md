# Saved-Evidence Commands

All commands ran from the assigned isolated `worktree` directory. No GPU execution,
network, installation, benchmark, autotune, history reads, commits or pushes occurred.
Only output artifacts were created. Python was exclusively the approved PDF-era
interpreter. Shell startup emitted an unrelated Lmod/Lua missing-posix diagnostic;
the read/extraction commands completed successfully.

Read `.agents/skills/tilebench-comparison/SKILL.md`, its `references/comparison.md`,
and sibling analysis references `tilebench-layout.md`, `nvidia-ncu.md`, `b200.md`,
and `diagnosis.md` with `sed`. Inspected supplied source/config, winner logs and CSV
with `cat`, `sed`, `rg`, and `tail`. No human reports or other trial files were read.

```bash
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python /scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py evidence/tilelang_fp32.ncu-rep --output output/tilelang_ncu.json
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python /scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py evidence/triton_fp32.ncu-rep --output output/triton_ncu.json
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python /scratch/arustagi/tilebench_pdf_env/bin/python tools/ncu_extract.py evidence/cutile_fp32.ncu-rep --output output/cutile_ncu.json
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python /scratch/arustagi/tilebench_pdf_env/bin/python output/analyze_saved.py
diff -u output/triton_source_0.py tilebench/benchmarks/operators/gaussian_blur/impl_triton.py
diff -u output/cutile_source_0.py tilebench/benchmarks/operators/gaussian_blur/impl_cutile.py
sed -n '1,90p' output/triton_sass.txt
sed -n '1,178p' output/cutile_sass.txt
sed -n '1,30p' output/tilelang_sass.txt
sed -n '900,975p' output/tilelang_sass.txt
```

Ad hoc approved-interpreter reads inspected exact scalar metric subsets, metric
inventories, NCU API documentation, source-file maps, opcode instances and explicit
ratios. `output/analyze_saved.py` makes the final extraction/provenance/evidence
generation reproducible, including hash assertions and kernel-AST comparison.
The SASS listings contain addresses inventoried by a PC-correlated sampling metric;
they are emitted-code evidence, not a static listing-size performance measurement.
