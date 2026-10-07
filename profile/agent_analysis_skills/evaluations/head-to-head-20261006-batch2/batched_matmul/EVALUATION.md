# Batched Matmul Paired Evaluation

## Verdict

Both skills produced useful, qualified explanations; no material diagnosis win
for TileBench. The original NCU arm extracted more specific wait-PC observations
for Triton/cuTile. The TileBench arm more directly explained the instruction-count
tradeoff, Triton's warmed pretranspose contract, and capture replay asymmetry.
These are complementary details, not evidence that one skill is generally better.

Original reports:
- [TileBench arm](tilebench/worktree/output/REPORT.md)
- [Original NCU arm](original_ncu/worktree/output/REPORT.md)

## Shared Findings

Both recover all three winner configurations independently: 128x128 output tiles,
TileLang K=64/stages=2, Triton K=32/stages=4, and cuTile K=64/occupancy hint=4.
Both correctly separate CSV latency (51.4/29.3/31.8 us for TL/TR/cuTile) from NCU
duration (48.576/32.064/34.848 us), and identify Torch as the fastest unprofiled
reference rather than assigning it an unsupported mechanism.

Both identify tensor-core underuse, not absence; all captures use UTCHMMA/TMEM.
They connect TileLang's lane-issued asynchronous copies and collective completion
waits to weaker operand delivery/dependency hiding than the peers' TMA paths.
They explicitly reject an entirely serial/no-pipeline interpretation. The leading
mechanism remains an inference because TileLang lacks collected stall counters.

Both also avoid the attractive but insufficient high-register-count explanation:
TileLang and Triton have the same three-CTA resource ceiling. Achieved occupancy
is not itself a cause. Small wave counts and cache/writeback/replay differences
remain alternatives. TileLang's measured HBM writes are below one logical output;
both decline to use that as proof of a traffic advantage or missing output work.

Triton/cuTile embedded kernel bodies match current source; TileLang's generated
CUDA content is absent. Both preserve that distinction rather than treating the
provided source revision as the capture revision. Neither claims a compiler defect
or a quantified optimization gain.

## Independent Checks

The 16 non-skill input files are identical between arms, report hashes match, and
no frozen inputs changed. Each arm's 5,206 full metric records were checked against
the raw API. Selected records: TileBench 80; original NCU 858, including PC instances.
Zero numeric/type/unit discrepancies and zero unidentified report records.
Record count is not a quality score and includes records without narrative claims.

TileBench's SASS inventories parse all 1,016 TileLang, 632 Triton and 1,120 cuTile
rows, with no warnings. The original arm uses raw SASS and PC records without a
saved parser inventory; no claim of equivalent parser validation is made.

One isolation limitation was self-disclosed by the original arm: an initial file
listing omitted `workdir` and listed parent workspace filenames. It reports reading
no parent file contents or answers, and records the incident in COMMANDS.md.
This is a real harness limitation: child tools inherit the parent's directory,
and the test relies on instructed boundaries rather than a filesystem sandbox.
The result is useful but not certified leak-proof. The second case's prompts remind
both arms equally to set explicit workdirs; no diagnostic hints were added.

## Limits

One trial per arm, artifacts and optional helpers supplied, no fresh collection,
no controlled kernel ablation. Good extraction does not prove causal attribution.
The observation is that both skills handle missing evidence well on this case,
not that TileBench's workflow materially improves general NCU diagnosis.
