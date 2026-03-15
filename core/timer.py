# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import glob
import json
import os
import re
import tempfile
import uuid
import warnings
from typing import Any, Callable

import torch

try:
    import triton.profiler as proton  # type: ignore
except Exception:
    proton = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _build_profile_base(kind: str, output_dir: str | None, label: str | None = None) -> str:
    base_dir = output_dir or tempfile.gettempdir()
    try:
        os.makedirs(base_dir, exist_ok=True)
    except Exception:
        warnings.warn(
            f"Failed to create proton output dir '{base_dir}', using temp dir.",
            RuntimeWarning,
        )
        base_dir = tempfile.gettempdir()
    # Sanitize label so it is safe as a filename component.
    safe_label = re.sub(r"[^a-zA-Z0-9_\-]", "_", label) if label else ""
    suffix = f"_{safe_label}" if safe_label else f"_{uuid.uuid4().hex[:8]}"
    return os.path.join(base_dir, f"tilebench_proton_{kind}{suffix}")


def _flush_l2_cache(flush_mb: int = 64) -> None:
    numel = max(1, flush_mb * 1024 * 1024 // 4)
    buf = torch.empty(numel, device="cuda", dtype=torch.float32)
    buf.fill_(1.0)
    torch.cuda.synchronize()


def _prepare_runner(
    f: Callable[..., Any],
    tuple_of_args: tuple[Any, ...],
    kwargs: dict[str, Any],
    use_cuda_graph: bool,
) -> Callable[[], Any]:
    if not use_cuda_graph:
        return lambda: f(*tuple_of_args, **kwargs)

    try:
        for _ in range(3):
            f(*tuple_of_args, **kwargs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = f(*tuple_of_args, **kwargs)

        def _graph_runner() -> Any:
            graph.replay()
            return static_output

        return _graph_runner
    except Exception:
        return lambda: f(*tuple_of_args, **kwargs)


def _load_profile_data(profile_base: str) -> tuple[Any, str]:
    for ext in (".json", ".hatchet"):
        path = profile_base + ext
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh), path
    for path in sorted(glob.glob(f"{profile_base}.*")):
        if os.path.isfile(path):
            with open(path) as fh:
                return json.load(fh), path
    raise FileNotFoundError(f"No profile file found for base: {profile_base}")


# ---------------------------------------------------------------------------
# Hatchet parsing
# ---------------------------------------------------------------------------

def _get_time_ns(metrics: Any) -> float:
    """Extract GPU time in nanoseconds from a Proton metrics dict."""
    if not isinstance(metrics, dict):
        return 0.0
    for key in ("time (ns)", "time(ns)", "time_ns"):
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    val = metrics.get("time (us)")
    if isinstance(val, (int, float)):
        return float(val) * 1e3
    val = metrics.get("time (ms)")
    if isinstance(val, (int, float)):
        return float(val) * 1e6
    return 0.0


def _collect_gpu_kernel_ns(node: Any) -> float:
    """
    Recursively sum total GPU kernel time (ns) from all descendant nodes
    that have device_type=CUDA/HIP metrics.

    Handles the extra `<captured_at>` level that Proton inserts when
    CUDA graph capture is used:

        launch (scope)   metrics: {}
        └── <captured_at>   metrics: {}
            └── kernel   count=100  time=160704 ns   ← actual data
    """
    if not isinstance(node, dict):
        return 0.0
    metrics = node.get("metrics", {})
    if isinstance(metrics, dict):
        dev = str(metrics.get("device_type", "")).upper()
        if dev in ("CUDA", "HIP"):
            ns = _get_time_ns(metrics)
            if ns > 0:
                return ns
    return sum(_collect_gpu_kernel_ns(child) for child in node.get("children", []))


def _find_scope_mean_ns(node: Any, scope_name: str, repeat: int) -> float:
    """
    Recursively search the hatchet tree for `scope_name` and return
    mean GPU kernel time (ns) = total_gpu_time / repeat.

    `repeat` is the authoritative denominator — the number of times run()
    was called inside the scope. This correctly handles operators that
    dispatch multiple kernels per call (each with its own Proton count):
    summing all kernel times and dividing by repeat gives the mean wall
    time of one complete run() invocation regardless of internal structure.

    Hatchet structure with repeat=N inside scope:

        ROOT
        ├── kernel  count=3           ← CUDA graph pre-warmup (outside scope)
        └── <scope_name>  metrics: {}
            └── <captured_at>  metrics: {}   ← may be absent without CUDA graph
                └── kernel  count=N  time=total_ns
    """
    if not isinstance(node, dict):
        return 0.0
    frame = node.get("frame", {})
    if isinstance(frame, dict) and frame.get("name") == scope_name:
        total_ns = _collect_gpu_kernel_ns(node)
        if total_ns > 0 and repeat > 0:
            return total_ns / repeat
        return 0.0
    for child in node.get("children", []):
        result = _find_scope_mean_ns(child, scope_name, repeat)
        if result > 0:
            return result
    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def report_benchmark(
    f: Callable[..., Any],
    tuple_of_args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    warmup: int = 20,
    repeat: int = 100,
    use_cuda_graph: bool = False,
    proton_scope_name: str = "launch",
    proton_context: str = "shadow",
    proton_backend: str | None = None,
    flush_l2: bool = False,
    keep_proton_files: bool = False,
    proton_output_dir: str | None = None,
    proton_file_label: str | None = None,
    # Kept for backward-compat call sites; unused
    quantiles: tuple[float, ...] = (),
    use_proton_scope: bool = True,
    proton_capture_scope_name: str = "graph",
) -> dict[str, float]:
    """
    Measure mean GPU kernel time using Proton (data="tree").

    Protocol
    --------
    - warmup  iterations run OUTSIDE the Proton session / scope
    - repeat  iterations run INSIDE  proton.scope(proton_scope_name)
    - Proton aggregates child kernel calls → hatchet node count = repeat
    - mean = total_time / count  (no per-sample distribution needed)

    Returns
    -------
    {"mean": float}   # milliseconds
    """
    del quantiles, use_proton_scope, proton_capture_scope_name  # unused

    if kwargs is None:
        kwargs = {}
    if proton is None:
        raise RuntimeError("triton.profiler (proton) is not available in this environment.")

    # --- warmup (outside Proton session so warmup kernels do not pollute tree) ---
    for _ in range(max(0, warmup)):
        if flush_l2:
            _flush_l2_cache()
        f(*tuple_of_args, **kwargs)
    torch.cuda.synchronize()

    # --- measurement session ---
    profile_base = _build_profile_base("tree", proton_output_dir, proton_file_label)
    session_id = proton.start(
        name=profile_base,
        context=proton_context,
        data="tree",
        backend=proton_backend,
    )
    try:
        # Build CUDA graph runner AFTER proton.start so graph capture is visible to Proton.
        runner = _prepare_runner(f, tuple_of_args, kwargs, use_cuda_graph=use_cuda_graph)
        torch.cuda.synchronize()
        for _ in range(max(1, repeat)):
            if flush_l2:
                # Flush L2 BEFORE entering scope so flush time is not measured by Proton.
                _flush_l2_cache()
            with proton.scope(proton_scope_name):
                runner()
        torch.cuda.synchronize()
    finally:
        proton.finalize(session=session_id)

    # --- parse hatchet ---
    hatchet_data, path = _load_profile_data(profile_base)
    if not keep_proton_files:
        try:
            os.remove(path)
        except OSError:
            pass

    roots = hatchet_data if isinstance(hatchet_data, list) else [hatchet_data]
    mean_ns = 0.0
    for root in roots:
        t = _find_scope_mean_ns(root, proton_scope_name, repeat)
        if t > 0:
            mean_ns = t
            break

    return {"mean": mean_ns / 1e6}
