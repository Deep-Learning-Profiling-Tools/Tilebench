"""TileBench++ Trn2 benchmark on the native PyTorch Neuron stack: native PyTorch
eager vs native NKI (diagnostics and smoke runs; the formal sweep is run_bench.py, whose
Neuron path is tilebench/core/neuron_native.py).

The xla_* modes (PyTorch/XLA) are legacy diagnostics, kept so archived runs that explained
the old stack's NKI host overhead still parse; native_torch_compiled is a torch.compile
diagnostic. They never enter a benchmark result
(schema.BENCHMARK_MODES, report.benchmark_records).

Nothing here imports torch_xla, torch_neuronx or nki at module import time; the
runtime adapters in ``runtime.py`` load their backend lazily, so the package imports
on CPU-only and GPU-only hosts.

Layout:
    schema.py     execution modes, status/fallback vocabularies, result records
    intervals.py  aggregation of device executions (sum / union / span)
    store.py      run directory (never overwrites), checkpoint/resume
    sources.py    per-operator NKI source manifest and pinned source overlays
    envinfo.py    environment snapshot
    bundles.py    shared input bundles with checksums
    runtime.py    native and (legacy) XLA runtime adapters
    worker.py     one (operator, case, stack) execution in a fresh process
    report.py     status table, pilot table, Markdown report (native; legacy XLA view separate)
"""
