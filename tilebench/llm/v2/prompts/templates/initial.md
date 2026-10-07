# API reference: {{dsl}} {{dsl_version}}

{{reference_skill}}

# Device context: {{device}}

{{device_context_skill}}

{{optimization_block}}

# Canonical algorithm contract: {{operator}}

{{algorithm_contract}}

# Task

- Operator: `{{operator}}`
- DSL: `{{dsl}}` {{dsl_version}}; output file: `{{output_file}}`
- Datatype: `{{dtype}}` (torch `{{torch_dtype}}`){{fp8_note}}
- Evaluated inputs: {{domain_block}}
- Numerical acceptance: `torch.testing.assert_close(output, reference, atol={{atol}}, rtol={{rtol}})`{{tolerance_note}}
- Interface: `{{run_signature}}`; returns {{returns}}.

## PyTorch functional reference (semantics only; not a performance target)

```python
{{functional_reference}}
```

## Output requirements

Return exactly one fenced block titled `{{output_file}}`. The file must define `run(...)` with the interface above and `get_last_config()`. Configuration values are literals or a deterministic function of the input shapes and static metadata. No autotuning, no runtime configuration search or timing, no table keyed to particular benchmark shapes, no cached results, no use of the reference implementation or other compute libraries for the operator's computation.
