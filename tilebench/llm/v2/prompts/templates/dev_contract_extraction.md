# Development-only: canonical algorithm extraction from one manual implementation

You are auditing a MANUAL kernel implementation to extract the algorithm it realizes. This session is a development tool with read access to human-written source; it is never used for kernel generation, and its output is reviewed by a human before any contract is approved.

Source language: {{source_dsl}}
Operator: {{operator}}
Source commit: {{source_sha}}

Read ONLY the files below. Do not consult the other DSL's implementation, TileLang/NKI implementations, autotune logs, benchmark results or profiler reports.

## impl_torch.py (functional semantics)

```python
{{torch_source}}
```

## impl_{{source_dsl}}.py (algorithmic approach)

```python
{{dsl_source}}
```

## config.yaml (shapes/dtypes only)

```yaml
{{config_excerpt}}
```

Produce a YAML document following the extraction schema exactly:

{{extraction_schema}}

Rules: cite line ranges; keep tunable values (tile sizes, warps, stages, occupancy, winners) ONLY in the audit-only fields; distinguish legality/mapping branches from algorithmic branches; write "unclear" rather than guessing; do not run code.
