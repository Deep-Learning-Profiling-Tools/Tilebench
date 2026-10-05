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

# Contract violation in round {{round}}, attempt {{attempt}} of {{max_attempts}}

Your candidate of this round was rejected before evaluation because it violates the task rules. It was not compiled, verified or timed, and it must not serve as the basis of a valid result.

## Violations found

{{violations}}

## Rejected candidate

```python title="{{output_file}}"
{{rejected_source}}
```

{{fallback_block}}

Produce a compliant implementation of the same task. This is attempt {{attempt}} of at most {{max_attempts}} generations in this round; if all generations of the round violate the contract, the round yields no result and its cost is still charged. Return exactly one fenced block titled `{{output_file}}`.
