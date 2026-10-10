"""Kernel-only timing of an operator whose run() copies an input and then updates the copy in
place (rope). The impl splits run() into restore() (refill the persistent buffer from the
input) and launch() (only the in-place kernel); every launch is preceded by

    restore() -> cache flush -> proton.scope { launch / graph replay of launch }

so the copy is neither timed nor left warm in the cache, and the kernel never runs on an
already-updated buffer. Otherwise this is tilebench.core.timer.report_benchmark: same warmup,
graph preparation and capture, Proton scope semantics and hatchet parsing, built from the
timer's own helpers. timer.py itself is left unchanged (the LLM v2 evaluator fingerprint pins
its bytes), and operators without the split keep using report_benchmark."""
from __future__ import annotations

import os
from typing import Any, Callable

import torch

from tilebench.core import timer


def _prepare_runner(launch: Callable[[], Any], restore: Callable[[], Any],
                    use_cuda_graph: bool) -> Callable[[], Any]:
    if not use_cuda_graph:
        return launch
    try:
        for _ in range(3):
            restore()
            launch()
        torch.cuda.synchronize()
        restore()                      # outside the capture: the graph holds launch() only
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        return graph.replay
    except Exception:
        return launch


def report_prepared_benchmark(
    launch: Callable[[], Any],
    restore: Callable[[], Any],
    *,
    warmup: int = timer.DEFAULT_WARMUP,
    repeat: int = timer.DEFAULT_REPEAT,
    use_cuda_graph: bool = False,
    proton_scope_name: str = "launch",
    proton_context: str = "shadow",
    proton_backend: str | None = None,
    flush_l2: bool = False,
    keep_proton_files: bool = False,
    proton_output_dir: str | None = None,
    proton_file_label: str | None = None,
) -> dict[str, float]:
    """Mean GPU kernel time (ms) of launch(), with restore() and the flush before every launch."""
    proton = timer.proton
    if proton is None:
        raise RuntimeError("triton.profiler (proton) is not available in this environment.")
    use_cuda_graph = timer.effective_use_cuda_graph(use_cuda_graph)

    # Warmup outside Proton session
    for _ in range(max(0, warmup)):
        restore()
        if flush_l2:
            timer._flush_l2_cache()
        launch()
    torch.cuda.synchronize()

    # Measurement session
    profile_base = timer._build_profile_base("tree", proton_output_dir, proton_file_label)
    session_id = proton.start(
        name=profile_base,
        context=proton_context,
        data="tree",
        backend=proton_backend,
    )
    try:
        runner = _prepare_runner(launch, restore, use_cuda_graph)
        torch.cuda.synchronize()
        for _ in range(max(1, repeat)):
            restore()
            if flush_l2:
                timer._flush_l2_cache()
            with proton.scope(proton_scope_name):
                runner()
        torch.cuda.synchronize()
    finally:
        proton.finalize(session=session_id)

    return {"mean": _scope_mean_ns(profile_base, proton_scope_name, repeat, keep_proton_files) / 1e6}


def _scope_mean_ns(profile_base: str, scope_name: str, repeat: int, keep_proton_files: bool) -> float:
    """Mean kernel time per timed launch (ns) inside scope_name, parsed as report_benchmark does."""
    hatchet_data, path = timer._load_profile_data(profile_base)
    if not keep_proton_files:
        try:
            os.remove(path)
        except OSError:
            pass
    roots = hatchet_data if isinstance(hatchet_data, list) else [hatchet_data]
    for root in roots:
        t = timer._find_scope_mean_ns(root, scope_name, repeat)
        if t > 0:
            return t
    raise RuntimeError(
        f"Failed to extract GPU timing from Proton hatchet data: "
        f"scope '{scope_name}' not found or reported zero time. "
        f"Profile file: {path}"
    )
