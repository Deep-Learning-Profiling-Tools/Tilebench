"""Latency measurement with explicit per-sample records.

Protocol: 1 warmup launch, then 3 timed launches; the statistic is their
arithmetic mean. Each timed launch is preceded by a last-level-cache eviction
OUTSIDE the timed scope (tilebench.core.timer._flush_l2_cache, 2x LLC) and
wrapped in its own Proton scope (`launch_1..3`) so that the three raw samples
are recoverable from the hatchet tree (core.timer.report_benchmark only
returns the mean of a single scope).

CUDA-graph policy follows tilebench.core.timer.effective_use_cuda_graph
(NVIDIA replays a graph, ROCm times eagerly). Unlike core.timer._prepare_runner,
a failed graph capture is NOT silently hidden: the record says
capture_succeeded=False and timing_execution_mode="eager", and the number of
eager preparation launches before capture is reported."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch

from tilebench.core import timer as core_timer

GRAPH_PREP_RUNS = 3   # eager launches before capture, as in core.timer._prepare_runner


@dataclass
class TimingRecord:
    warmup_runs: int
    timed_runs: int
    samples_ms: list[float]
    mean_ms: float | None
    requested_use_cuda_graph: bool
    effective_use_cuda_graph: bool
    capture_succeeded: bool | None       # None when no capture was attempted
    capture_error: str | None
    graph_prep_runs: int                 # eager runs consumed by capture preparation
    timing_execution_mode: str           # graph | eager
    flush_before_each_launch: bool
    flush_buffer_mb: int | None
    scope_names: list[str] = field(default_factory=list)
    note: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _capture(f: Callable[[], Any]) -> tuple[Callable[[], Any] | None, str | None]:
    try:
        for _ in range(GRAPH_PREP_RUNS):
            f()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = f()
        torch.cuda.synchronize()

        def replay():
            graph.replay()
            return static_out
        return replay, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def measure(f: Callable[[], Any], *, warmup: int = 1, repeat: int = 3, use_cuda_graph: bool = True,
            flush: bool = True, before_launch: Callable[[], None] | None = None,
            proton_output_dir: str | None = None, label: str | None = None) -> TimingRecord:
    """f() runs the whole operator on fixed inputs. before_launch() restores
    mutated inputs (outside the timed scope) when the contract requires it."""
    proton = core_timer.proton
    if proton is None:
        raise RuntimeError("triton.profiler (proton) is not available")
    effective = core_timer.effective_use_cuda_graph(use_cuda_graph)
    flush_mb = core_timer._flush_l2_buffer_mb() if flush else None

    def pre():
        if before_launch is not None:
            before_launch()
        if flush:
            core_timer._flush_l2_cache()

    for _ in range(max(0, warmup)):
        pre()
        f()
    torch.cuda.synchronize()

    runner, capture_error, prep = f, None, 0
    captured: bool | None = None
    if effective:
        pre()
        runner, capture_error = _capture(f)
        prep = GRAPH_PREP_RUNS + 1
        captured = runner is not None
        if runner is None:
            runner = f

    base = core_timer._build_profile_base("tree", proton_output_dir, label)
    session = proton.start(name=base, context="shadow", data="tree", backend=None)
    scopes = [f"launch_{i + 1}" for i in range(max(1, repeat))]
    try:
        for name in scopes:
            pre()
            torch.cuda.synchronize()
            with proton.scope(name):
                runner()
        torch.cuda.synchronize()
    finally:
        proton.finalize(session=session)
    data, path = core_timer._load_profile_data(base)
    try:
        import os
        os.remove(path)
    except OSError:
        pass
    roots = data if isinstance(data, list) else [data]
    samples = []
    for name in scopes:
        ns = 0.0
        for root in roots:
            ns = core_timer._find_scope_mean_ns(root, name, 1)
            if ns > 0:
                break
        samples.append(ns / 1e6)
    mean = sum(samples) / len(samples) if all(s > 0 for s in samples) else None
    return TimingRecord(warmup_runs=warmup, timed_runs=repeat, samples_ms=samples, mean_ms=mean,
                        requested_use_cuda_graph=bool(use_cuda_graph), effective_use_cuda_graph=effective and bool(captured),
                        capture_succeeded=captured, capture_error=capture_error, graph_prep_runs=prep,
                        timing_execution_mode="graph" if (effective and captured) else "eager",
                        flush_before_each_launch=flush, flush_buffer_mb=flush_mb, scope_names=scopes,
                        note=None if samples and all(s > 0 for s in samples) else "a scope reported zero time")
