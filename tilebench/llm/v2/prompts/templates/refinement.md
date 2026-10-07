# API reference: {{dsl}} {{dsl_version}}

{{reference_skill}}

# Device context: {{device}}

{{device_context_skill}}

{{optimization_block}}

# Canonical algorithm contract: {{operator}}

{{algorithm_contract}}

# Task (unchanged)

- Operator: `{{operator}}`; DSL `{{dsl}}` {{dsl_version}}; output file `{{output_file}}`
- Datatype: `{{dtype}}` (torch `{{torch_dtype}}`){{fp8_note}}
- Evaluated inputs: {{domain_block}}
- Numerical acceptance: `atol={{atol}}, rtol={{rtol}}`{{tolerance_note}}
- Interface: `{{run_signature}}`; returns {{returns}}.

## PyTorch functional reference

```python
{{functional_reference}}
```

# Optimization round {{round}} of {{rounds}}

{{prev_block}}
{{best_valid_block}}
## Runtime history of this task (valid candidates only: all {{n_cases}} cases valid; geometric mean over the {{n_cases}} cases of the per-case runtime, each the mean of 3 timed runs after 1 warmup)

{{runtime_history}}

Improve on the best valid geometric-mean runtime while keeping the contract and staying correct on every case. Return exactly one fenced block titled `{{output_file}}`.
