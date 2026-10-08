# Reconciliation guide: from two extractions to a canonical contract

Source commit (S_main): `ea04fb368c88ed4ee8621e3b1b1d6013a96f1cfe`.

Inputs per operator `<op>`:
- `docs/llm_v2/canonical/extraction/triton/<op>.yaml`
- `docs/llm_v2/canonical/extraction/cutile/<op>.yaml`
- the sources, to VERIFY every claim before it enters a contract:
  `tilebench/benchmarks/operators/<op>/{impl_torch.py,impl_triton.py,impl_cutile.py,config.yaml}`
  and the generator in `tilebench/data/tensors.py`.
- Do NOT read `impl_tilelang.py`, `impl_nki.py`, results, autotune logs or NCU data.

Outputs per operator, under `tilebench/llm/v2/contracts/data/<op>/`:

## 1. `audit.json` (non-prompt provenance)

Exactly the shape documented in `tilebench/llm/v2/contracts/schema.py`
(`AUDIT_SCHEMA = "tilebench-contract-audit/1"`). Required keys:
`schema, operator, source_sha, contract_revision (1), status, approved_by (null),
approved_on (null), sources, aspects, fq_boundary, human_reference_comparable,
review_items, contract_sha256 (null)`.

- `sources`: `{"triton": {"file": ..., "functions": [{"name", "lines": [a,b]}]},
  "cutile": {...}, "torch": {...}}` — copy from the extractions, verify by reading.
- `aspects`: ALL nine of `functional_semantics, stages, dependencies,
  reduction_scan_sort, precision, preprocessing, intermediate_storage, mutation,
  hardware_dispatch`, each `{"status": aligned|mapping-only|needs-review,
  "triton": <one-line fact with line cite>, "cutile": <one-line fact with line cite>,
  "note": <why this status>}`.
  - `aligned`: same algorithmic content.
  - `mapping-only`: differs only in tiling/layout/launch/pipelining/primitive
    choice/legality fallback; the contract must not pin either choice.
  - `needs-review`: a genuine algorithmic or boundary divergence (e.g. one side
    caches a prepacked operand across calls, one side counts a different
    reduction structure, different precision path). Never resolve it by
    intersection or by picking the faster side; state the exact question.
- `fq_boundary`: `{"bytes_expr", "flops_expr", "consistent": "true"|"false"|"unclear",
  "note"}` — a boundary mismatch (uncounted in-run clone/zero-fill/scratch;
  untimed cached prepack; dense flops for causal) makes `consistent` false or
  unclear and MUST produce a review item.
- `human_reference_comparable`: false when the manual timed boundary differs
  from what the contract will require of a generated implementation (e.g. the
  manual kernel benefits from an untimed cached transpose).
- `status`: `draft` when every aspect is aligned/mapping-only and no review
  item blocks the contract; `needs-review` otherwise. NEVER `approved`.
- `review_items`: one sentence each, specific, with the decision a human must make.

## 2. `contract.md` (model-visible; the ONLY one of the three the generator sees)

DSL-neutral English Markdown, 2-8 KB, with these sections in this order:

```
# <op>: canonical algorithm contract
## Functional semantics
## Inputs and outputs            (shapes symbolic, dtypes, aliasing, mutation rules)
## Required logical stages       (numbered; dependencies; what may be fused/split/reordered)
## Algorithm family and structure (reduction/scan/sort constraints, tie rules, masking values)
## Precision and accumulation
## Preprocessing and timing boundary (what run() must do itself; prepacked-input rules)
## Permitted implementation mappings
## Forbidden substitutions
## Permitted PyTorch operations  (explicit list: e.g. torch.empty / empty_like for outputs,
                                  declared scratch buffers, .view/.reshape/.contiguous only
                                  where listed, dtype metadata; everything else forbidden)
## Open review items             (ONLY if audit.status is needs-review; one line each,
                                  phrased neutrally; no manual-kernel details)
```

Hard rules for contract.md (checked by `tilebench.llm.v2.contracts.loader.leak_check`):
- No tile sizes, block sizes, warp counts, stage counts, occupancy values,
  autotune winners, latencies (no `ms`/`us` figures), speedups, "winner",
  "best config", T_SOL, SOL efficiency, no kernel source, no `def run(`,
  no `@triton.jit` / `@ct.kernel`. Use "logical output tile" / "local
  accumulator" / "one program per row", never "warp"/"CTA"/"TMEM" as a
  requirement.
- Logical stages are not kernel counts: say explicitly which stages may be
  fused into one launch or split across launches, based on dependencies.
- Describe input mutation precisely: if neither manual implementation mutates
  inputs, the contract forbids mutation. If the reference output is a view of
  an in-run buffer, say what the generated run() must return.
- Where the manual implementations diverge (needs-review), the contract
  states the semantics both agree on and lists the open question under
  "Open review items" without revealing either implementation's tuning.
- Never mention Triton, cuTile, "manual", "human", "the reference kernel".

## 3. `evaluator_rules.json` (evaluator-only)

`RULES_SCHEMA = "tilebench-evaluator-rules/1"`, keys:
`schema, operator, contract_revision (1), required_stages
[{"id","description"}], forbidden_substitutions [{"pattern": <python regex>,
"message", "level": "confirmed"|"suspicious"}], required_evidence
[{"any_of": [<regex>...], "message", "level": "suspicious"}],
allowed_torch_calls [dotted names, e.g. "torch.empty", "torch.empty_like",
"torch.zeros" (only if a zero-filled scratch is part of the algorithm)],
mutation {"inputs_mutated": [names], "restore_required": bool}, outputs
{"structure": "tensor"|"tuple"|"list", "count": int, "aliasing": str},
timing_boundary {"includes_preprocessing": bool, "notes": str},
tolerance_source: "config.verify"`.

- `forbidden_substitutions` level `confirmed` only for unambiguous
  replacements of the whole computation (e.g. `torch\.sort|torch\.argsort`
  in a sort operator, `torch\.(nn\.functional\.)?conv2d` in 2d_conv,
  `torch\.topk` in top_k_selection, `torch\.histc|torch\.bincount`).
  Everything that needs judgement is `suspicious`.
- `required_evidence` is advisory (suspicious): e.g. for a reduction operator
  `any_of: ["tl\\.sum", "ct\\.sum", "T\\.reduce_sum", "reduce"]`. Keep it
  DSL-agnostic by listing the Triton, cuTile, TileLang and NKI spellings.
- `allowed_torch_calls` must be minimal and consistent with contract.md.

## Verification before you finish

For each operator run:
```
python -c "import json,sys; sys.path.insert(0,'.'); from tilebench.llm.v2.contracts.loader import load_contract; c=load_contract('<op>', require_approved=False); print(c.status, c.sha256[:8])"
```
It must print the status without raising (schema and leak checks pass).
