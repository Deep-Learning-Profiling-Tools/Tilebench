# Second Matched Skill Batch

Four independent GPT-6.1 Sol runs, high reasoning, sequential, with frozen skills
from `ff08a878`. Both arms compare TileLang, Triton and cuTile on each case.
Source, benchmark CSV, winner logs, raw reports and optional extraction tooling
are identical within each pair. No human analyses were supplied or GPU runs made.

## Original Reports

| Case | TileBench comparison skill | Original NCU report skill |
|---|---|---|
| Batched matmul FP16, BATCH=32, M=N=K=640 | [Report](batched_matmul/tilebench/worktree/output/REPORT.md) | [Report](batched_matmul/original_ncu/worktree/output/REPORT.md) |
| Gaussian blur FP32, input_rows=10240, 7x7 | [Report](gaussian_blur/tilebench/worktree/output/REPORT.md) | [Report](gaussian_blur/original_ncu/worktree/output/REPORT.md) |

See the [summary](SUMMARY.md), [matmul evaluation](batched_matmul/EVALUATION.md),
[blur evaluation](gaussian_blur/EVALUATION.md) and [predeclared protocol](PROTOCOL.md).
The first pair is published [alongside this batch](../head-to-head-20261006/README.md).
Together these bundles contain six distinct original head-to-head agent reports.

Original reports and supporting outputs are preserved unchanged. Their checksums
are in `report_checksums.json`. Statements in the original summary about being
uncommitted describe its pre-publication state, not this published bundle.

## Scope and Reproduction

The bundle includes exact metric records, source/SASS extracts, scripts, winner
logs, input manifests and parent audits. It does not include raw `.ncu-rep` files,
complete repository checkouts, caches or credentials. Each arm's
`worktree/evidence/reports.json` identifies the saved files, checksums and immutable
revision of [bcui2/NCU_report](https://huggingface.co/datasets/bcui2/NCU_report).
The tested skills are in the repository's `skills/` directory; attribution for the
unchanged third-party NCU skill is in its README and the first pair's provenance.

For replay, restore the source/CSV revision in `trial.json`, frozen skill/tooling
files, and raw reports at the recorded checksums. Relocate original local paths
in commands as necessary. Input hashes distinguish the study inputs from the
revision containing this publication. Parent audits establish extraction accuracy,
not unique causal explanations or certified filesystem isolation.

No skill diagnosis advantage was demonstrated on these samples. The original NCU
arm adds useful PC-level attribution, while TileBench-specific discovery remains
outside the supplied-artifact experiment. Limitations, including a parent-filename
listing incident and derived API unit-label discrepancies, are retained in the
evaluations rather than removed from the published record.
