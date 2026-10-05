"""TileBench++ LLM experiment protocol, revision 2.

This package is the v2 entry point (``python -m tilebench.llm.v2``). The legacy
pipeline in ``tilebench.llm`` (joint Triton/cuTile generation, roofline feedback,
``benchmarks/llm_generated`` outputs) is kept as-is and never written to by v2.

Sub-packages, by responsibility:

- ``manifests``      frozen study configuration, schemas and validation
- ``tasks``          task identity, representative-case selection, eligibility
- ``contracts``      canonical algorithm contracts (model-visible / evaluator / audit)
- ``skills``         manifest-driven, hash-pinned context loading
- ``prompts``        English prompt templates and the deterministic renderer
- ``providers``      GPT / Claude / mock adapters, usage normalization, ledger
- ``validation``     strict response parser, static evidence, contract checks
- ``evaluation``     isolated worker, 1-warmup/3-timed measurement, anti-cache
- ``metrics``        T_SOL, cumulative cost, SOL-Efficiency@B, aggregation
- ``orchestration``  ten-round state machine, persistence, resume, locks
- ``distillation``   source-device / fold-scoped evidence access and synthesis
"""

PROTOCOL = "tilebench-llm-v2"
PROTOCOL_REVISION = 2
