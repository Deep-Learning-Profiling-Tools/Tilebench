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
- Fixed input case:
{{params_block}}
- Numerical acceptance: `atol={{atol}}, rtol={{rtol}}`{{tolerance_note}}
- Interface: `{{run_signature}}`; returns {{returns}}.

## PyTorch functional reference

```python
{{functional_reference}}
```

# Optimization round {{round}} of {{rounds}}

{{prev_block}}
{{best_valid_block}}
## Runtime history of this task (valid candidates only; ms, mean of 3 timed runs after 1 warmup)

{{runtime_history}}

Improve on the best valid runtime while keeping the contract. Return exactly one fenced block titled `{{output_file}}`.
