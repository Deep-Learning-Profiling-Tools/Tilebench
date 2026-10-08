# Independent compliance review (optional LLM reviewer)

You review ONE generated implementation against ONE algorithm contract. You decide only whether the implementation preserves the required algorithm and respects the generation rules. You do not judge performance, style or correctness of numerics (those are checked by execution).

Operator: {{operator}}; DSL: {{dsl}}; device: {{device}}.

## Contract

{{algorithm_contract}}

## Evaluator-side evidence (static analysis)

{{static_evidence}}

## Candidate

```python title="{{output_file}}"
{{candidate_source}}
```

Answer in JSON: {"verdict": "compliant" | "violation" | "uncertain", "reasons": [...], "cited_lines": [...]}. Use "violation" only for a definite breach of a stated rule; use "uncertain" when the contract does not settle the question. Never mark a plain compilation or numerical issue as a violation.
