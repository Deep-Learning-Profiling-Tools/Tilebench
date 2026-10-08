"""Empirical Roofline calibration for the TileBench++ LLM protocol.

Task-independent probes measure sustained HBM bandwidth and compute rates
per explicitly declared arithmetic mode on the actual device/scope. The
result is an *empirical profile* (schema.PROFILE_SCHEMA), written to its own
calibration directory; it never overwrites the legacy peak table
(tilebench/data/peak_performance/<GPU>.json) or any benchmark result.

    schema.py       mode registry, profile schema, validation, hashing
    protocol.py     the versioned measurement protocol and its hash
    stats.py        batch statistics, point validity, mode selection rules
    environment.py  device/toolchain/clock/precision-flag capture
    gpu_probes.py   CUDA/ROCm probes (torch BLAS GEMMs, copy/fill, Triton
                    streaming and FMA micro-benchmarks)
    run.py          orchestration under the device lock + artifact writer
"""
