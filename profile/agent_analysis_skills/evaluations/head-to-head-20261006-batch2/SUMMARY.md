# Second Head-to-Head Batch

Completed four fresh GPT-6.1 Sol runs, high reasoning, sequential, on two additional
B200 cases across TileLang, Triton and cuTile. Skills unchanged from `ff08a878`.
Saved source, CSV, winner logs, NCU reports and extraction tools were identical
within each pair; no human answers were provided and no GPU work was performed.

| Case | Result | Parent assessment |
|---|---|---|
| Batched matmul FP16, batch 32, 640 cubed | Both recover configs, tensor-core underuse, copy/completion schedule and evidence limits | Comparable core diagnosis; complementary details |
| Gaussian blur FP32, 10240 square, 7x7 | Both recover configs and explain on-chip work versus register-residency tradeoff | Original NCU adds stronger PC attribution and catches a rule-engine false warning |

Together with the first max-pooling pair, none of these three cases demonstrates
materially better diagnosis from our TileBench skill. Original NCU is at least
competitive and repeatedly supplies useful source/PC attribution. TileBench's
artifact discovery and integration advantages are not tested by this supplied-input
experiment. A useful repository integration is not the same as a novel diagnosis
method; there is no evidence here supporting the latter claim.

## Original Agent Reports

- [Matmul, TileBench](batched_matmul/tilebench/worktree/output/REPORT.md)
- [Matmul, original NCU](batched_matmul/original_ncu/worktree/output/REPORT.md)
- [Blur, TileBench](gaussian_blur/tilebench/worktree/output/REPORT.md)
- [Blur, original NCU](gaussian_blur/original_ncu/worktree/output/REPORT.md)
- [Matmul evaluation](batched_matmul/EVALUATION.md)
- [Blur evaluation](gaussian_blur/EVALUATION.md)
- [Protocol](PROTOCOL.md)

All numeric records audited against the raw API match; no inputs changed and all
report hashes match. Blur original NCU has three derived-unit label discrepancies
on reload, documented separately from numerical accuracy. Original matmul briefly
listed parent filenames due to an omitted workdir; no parent contents were reported
read, but the boundary is not certified leak-proof. Full limitations are in the
per-case evaluations. More metric records or longer reports are not scored as wins.

No skills were changed, and this batch has not been committed or pushed.
