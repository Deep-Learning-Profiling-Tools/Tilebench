# Saved-Evidence Skill Comparison

Two independent GPT-6.1 Sol agents, high reasoning, analyzed the same B200 FP32
maximum-shape max-pooling case across TileLang, Triton and cuTile. They ran
sequentially with fresh context and identical source, CSV, winning configuration
logs, saved NCU reports and optional extraction helpers. No GPU work was run.

## Read the Results

- [TileBench-skill agent report](tilebench/worktree/output/REPORT.md)
- [Original NCU-skill agent report](original_ncu/worktree/output/REPORT.md)
- [Parent evaluation](EVALUATION.md)
- [Predeclared protocol](PROTOCOL.md)

Original agent reports, scripts and extracted evidence are preserved unchanged.
The parent audit initially needed opcode-instance support; the saved final audits
check instances at their actual index and correlation ID. Both arms have zero
numeric mismatches and complete SASS parsing. Derived API unit-label differences
remain recorded rather than silently corrected.

## Skills and Attribution

The complete tested TileBench skills are in
[`skills/tilebench-analysis`](../../../../skills/tilebench-analysis/SKILL.md) and
[`skills/tilebench-comparison`](../../../../skills/tilebench-comparison/SKILL.md).
Install both together; the comparison skill uses the analysis skill's shared resources.

The unchanged installed original skill is in
[`skills/ncu-report-skill`](../../../../skills/ncu-report-skill/SKILL.md).
It is third-party work, not a TileBench contribution. Its upstream README links
[MIT Han Lab's Kernel Design Agents](https://github.com/mit-han-lab/kernel-design-agents)
and the standalone [DongyunZou/ncu-report-skill repository](https://github.com/DongyunZou/ncu-report-skill).
The README declares MIT licensing; the installed snapshot did not include a
separate LICENSE file. We do not invent copyright holders or an upstream revision.
`ncu-skill-provenance.json` records every copied upstream file's checksum.

## Artifact Boundaries

This publication contains no raw `.ncu-rep` files, nested Git repositories or
credentials. Each arm's `worktree/evidence/reports.json` records the exact public
Hugging Face dataset revision, report filename and checksum. Winner logs are
included; benchmark/source revisions and frozen input hashes are in `trial.json`.
The raw profiles remain in [bcui2/NCU_report](https://huggingface.co/datasets/bcui2/NCU_report).
Use the TileBench skill's `hf_ncu_report.py` downloader and validate the recorded
hashes before reproducing. Some original commands and provenance refer to the
local trial path; relocate those paths when replaying elsewhere. The three raw
report files must be restored into each arm's `worktree/evidence/` first.

This is one trial per arm, not proof of general superiority. With artifacts and
helpers already supplied, our skill did not show a material diagnosis advantage.
TileBench navigation and reproducible packaging were not independently tested.
