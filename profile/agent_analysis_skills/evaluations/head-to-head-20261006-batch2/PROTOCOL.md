# Second Head-to-Head Batch

Freeze the skills at published commit `ff08a878`; do not revise prompts or skills
in response to an arm's findings during a matched pair. GPT-6.1 Sol, high reasoning,
fresh context, sequential arms. No GPU capture, benchmark or autotuning.

Cases, all on B200 and comparing TileLang/Triton/cuTile:

1. Batched matmul FP16, BATCH=32, M=N=K=640: original NCU skill first, TileBench second.
2. Gaussian blur FP32, maximum H=W=10240: TileBench first, original NCU skill second.

Identical supplied inputs within each pair: pinned source/CSV at `17d2d4f6`,
winner logs at `9455c0bd`, saved reports at dataset revision `21037737`, and
the same optional extraction scripts. Neither agent may read human narratives,
prior agents' results, another arm, or historical Git answers. Isolation is by
instruction, not a security boundary. Source/capture linkage must be assessed.

Criteria unchanged from the first pair: case/config recovery, correct separate
CSV/NCU ratios, exact metric/action identities, code/config-to-counter explanation,
nonproportional instruction/timing reasoning, supported causal claims and alternatives,
honest missing-evidence/provenance handling, and complete SASS parsing if used.
Do not count more metrics or more text as inherently better diagnosis.

Preserve original reports. Audit both against raw reports after each pair. Record
numeric discrepancies separately from reload-dependent derived unit labels and
per-opcode instance normalization. Compare both successful and unsuccessful paths.
One run per arm per case remains insufficient for a general reliability claim.
Collection and artifact-discovery advantages remain outside this supplied-input test.

No automatic commit or push: this batch is separate from the already published pair.
