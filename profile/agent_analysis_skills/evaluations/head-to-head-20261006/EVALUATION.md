# Head-to-Head Result

## Verdict

No material diagnosis advantage for our skill on this initial case. Both fresh
GPT-6.1 Sol agents recovered the selected configurations, connected implementation
and SASS differences to the counters, and qualified their conclusions. The
original NCU-skill arm additionally recovered per-PC/source-line stall samples and
NCU rule outputs. Our arm produced a more explicitly structured comparison and
coverage artifact, not a demonstrably better performance explanation.

This does not establish equivalence across kernels. It is one paired trial with
shared extraction helpers and supplied artifacts, so it does not test the value
of TileBench-specific navigation or the original skill's capture workflow.

## Original Reports

- Our skill: [REPORT.md](tilebench/worktree/output/REPORT.md).
- Installed original NCU skill: [REPORT.md](original_ncu/worktree/output/REPORT.md).
- Frozen design and limitations: [PROTOCOL.md](PROTOCOL.md).
- Input manifests and independent audits: each arm's `trial.json` and `audit.json`.

The reports are preserved as written. Neither agent received the human analysis,
previous agent answers, or the other arm's output. Isolation was instructed rather
than filesystem-enforced; no claim of a security-enforced blind test is made.

## Criteria Comparison

| Criterion | TileBench skill | Original NCU skill |
|---|---|---|
| Correct case and all three winner configs | Yes | Yes |
| CSV versus NCU timings and ratio direction | Correct and separate | Correct and separate |
| Exact selected NCU numeric records | 202; no discrepancies | 126; no discrepancies |
| Code/configuration connected to counters | Yes | Yes |
| Nonproportional instructions versus latency | Explained | Explained |
| Causal limits and alternative mechanisms | Explicit tradeoffs and alternatives | Mixed-cost diagnosis and qualified recommendations |
| Capture/source provenance | TR/cuTile kernel AST matches; TL inferred | Same |
| SASS completeness | 112 TL / 264 TR / 488 cuTile | Same, all rows parsed |
| Additional diagnostic work | Opcode instances, comparison/coverage JSON | Per-PC stall hotspots, rule-engine outputs |

## Shared Findings

CSV latency is TileLang 206.3 us, Triton 225.7 us, cuTile 299.5 us. Both avoid
overstating TileLang's 8.6% advantage over Triton, which falls inside the study's
10% parity band. Saved NCU duration is separately 211.712 / 227.520 / 306.432 us.

Both find much shorter TileLang indexing/predicate work, no shared-memory output
conversion in TileLang, and shared conversion plus vector stores in the other two.
TileLang and Triton have identical global-load request/sector counts and nearly
identical HBM traffic, so lower instruction count does not mean lower input traffic
or proportional latency savings. cuTile has 64 registers/thread versus 37/39,
50% theoretical occupancy versus 75%, 12% more global-load sectors and about 10%
more HBM reads. Both identify multiple compatible contributors rather than
assigning unsupported fractions of latency to individual causes.

The original arm provides more explicit load-dependent stall attribution. Our arm
more directly highlights the vector-store benefit and why bank conflicts or low
occupancy alone do not explain the ordering. Neither observation establishes a
controlled causal ablation. In particular, reduced theoretical occupancy does not
prove actual scheduler starvation, and overlap of integer work with memory service
remains an interpretation rather than a measured decomposition.

## Independent Audit

All 16 frozen non-skill inputs are byte-identical between arms. No frozen inputs
changed, all raw report hashes match, and no metric records lack report identity.
Each arm's 7,365 full scalar records and all selected records have zero numeric or
type mismatches against the raw API. Our opcode-instance records were also checked
at their instance indexes and correlation IDs, not against aggregate values.

Both full dumps have the same 36 derived-metric unit-label differences on reload
(12 per report); none occur in the selected evidence. These are retained in the
audit rather than counted as numeric extraction failures or silently corrected.
They reinforce that derived API unit labels need care. SASS parsing is complete
in both arms, with no warnings. This audit checks extraction, not causal truth.

## Implication

The useful differentiation currently looks like TileBench artifact/configuration
navigation and reproducible evidence packaging, not exclusive NCU diagnosis
capability. Keep that claim modest. A harder matched case such as Gaussian blur
or attention, followed by repetitions, would test whether the diagnosis workflow
adds value beyond this relatively readable max-pooling case. No skills were edited,
no GPU work was run, and nothing was committed or pushed during this pair.
