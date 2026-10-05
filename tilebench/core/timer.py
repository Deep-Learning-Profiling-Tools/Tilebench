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

from tilebench.hardware import last_level_cache_bytes

try:
    import triton.profiler as proton  # type: ignore
except ImportError:
    proton = None


def _build_profile_base(kind: str, output_dir: str | None, label: str | None = None) -> str:
    base_dir = output_dir or tempfile.gettempdir()
    try:
        os.makedirs(base_dir, exist_ok=True)
    except OSError:
        warnings.warn(
            f"Failed to create proton output dir '{base_dir}', using temp dir.",
            RuntimeWarning,
        )
        base_dir = tempfile.gettempdir()
    safe_label = re.sub(r"[^a-zA-Z0-9_\-]", "_", label) if label else ""
    suffix = f"_{safe_label}" if safe_label else f"_{uuid.uuid4().hex[:8]}"
    return os.path.join(base_dir, f"tilebench_proton_{kind}{suffix}")


#: Formal measurement protocol when neither config.yaml nor the CLI sets one:
#: warmup launches, then timed launches whose kernel time is averaged.
DEFAULT_WARMUP = 1
DEFAULT_REPEAT = 3


#: Why a requested CUDA-graph measurement runs eagerly on ROCm.
HIP_GRAPH_TIMING_NOTE = (
    "ROCm HIP Graph timing fallback: Proton/roctracer does not reliably attribute "
    "child kernels in graph replay")


def effective_use_cuda_graph(requested: bool) -> bool:
    """Whether a measurement that requests CUDA-graph replay actually uses it.

    The config's use_cuda_graph is the requested behaviour; the runtime decides
    the executed one. On ROCm (a HIP build of torch) the timed region always
    runs eagerly: with Proton's roctracer backend a HIP Graph replay is
    recorded as its first child kernel only (bitonic_sort: 1 of 191 kernels
    per replay on MI300X), so graph timing would silently under-report
    multi-kernel operators. NVIDIA keeps graph replay. Decided by the torch
    build, never by the --gpu result label."""
    return bool(requested) and not torch.version.hip


def timing_mode(requested: bool) -> dict[str, Any]:
    """Requested vs. effective execution mode of the timed region, for logs."""
    effective = effective_use_cuda_graph(requested)
    return {
        "requested_use_cuda_graph": bool(requested),
        "effective_use_cuda_graph": effective,
        "timing_execution_mode": "graph" if effective else "eager",
        "timing_note": HIP_GRAPH_TIMING_NOTE if requested and not effective else None,
    }


# Pre-allocated L2 flush buffer (lazily initialized on first use).
_l2_flush_buf: torch.Tensor | None = None


def _flush_l2_buffer_mb() -> int:
    """Eviction buffer size: 2x the device's last-level cache, queried at runtime.

    The cache is the one tilebench.hardware.last_level_cache_bytes() reports:
    the runtime L2 size on NVIDIA (so B200 still evicts 2x its 126.5 MB L2), or
    a measured device-level LLC for architectures whose L2 is not the last
    level (CDNA3: the 256 MiB Infinity Cache, so MI300X evicts 512 MiB, not 2x
    its 4 MiB L2); an architecture in hardware._LLC_CALIBRATION_REQUIRED
    without a validated size raises UncalibratedCacheError, so a flushed
    measurement fails instead of evicting only the L2. Hardcoding a size silently under-evicts on GPUs with a larger cache
    than the author assumed (the previous 64 MB constant covered only half of
    B200's 126.5 MB L2, leaving small-input operators warm).

    The 2x factor is measured, not assumed: on B200 a 64/128/256/512 MB sweep
    showed 64 MB leaves inputs warm (flash_decode 26.5 -> 45.0 us,
    cross_entropy 7.0 -> 7.9 us when raised), while 256 MB (~2x L2) and
    512 MB agree within 0.2% — past ~2x L2 a larger buffer no longer changes
    the cold-entry measurement. Intra-operator producer-consumer reuse is
    unaffected (linear_self_attention is flat across all four sizes).
    """
    llc_bytes = last_level_cache_bytes()
    if llc_bytes is None:
        llc_bytes = 128 * 1024 * 1024     # conservative fallback
    return max(64, 2 * llc_bytes // (1024 * 1024))


def _flush_l2_cache(flush_mb: int | None = None) -> None:
    """Evict the device's last-level cache at the operator boundary (never
    between an operator's kernels)."""
    global _l2_flush_buf
    if flush_mb is None:
        flush_mb = _flush_l2_buffer_mb()
    numel = max(1, flush_mb * 1024 * 1024 // 4)
    if _l2_flush_buf is None or _l2_flush_buf.numel() != numel:
        _l2_flush_buf = torch.empty(numel, device="cuda", dtype=torch.float32)
    _l2_flush_buf.fill_(1.0)
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
        try:
            with open(path) as fh:
                return json.load(fh), path
        except FileNotFoundError:
            continue
    for path in sorted(glob.glob(f"{profile_base}.*")):
        if os.path.isfile(path):
            with open(path) as fh:
                return json.load(fh), path
    raise FileNotFoundError(f"No profile file found for base: {profile_base}")


def _get_time_ns(metrics: Any) -> float:
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
    """Sum GPU kernel time (ns) from descendant nodes with device_type=CUDA/HIP."""
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
    """Find scope_name in hatchet tree and return mean GPU kernel time (ns)."""
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


def report_benchmark(
    f: Callable[..., Any],
    tuple_of_args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    use_cuda_graph: bool = False,
    proton_scope_name: str = "launch",
    proton_context: str = "shadow",
    proton_backend: str | None = None,
    flush_l2: bool = False,
    keep_proton_files: bool = False,
    proton_output_dir: str | None = None,
    proton_file_label: str | None = None,
) -> dict[str, float]:
    """Measure mean GPU kernel time (ms) using Proton (data="tree").

    use_cuda_graph is the requested mode; effective_use_cuda_graph() decides
    the executed one (eager on ROCm, before any graph is captured)."""
    if kwargs is None:
        kwargs = {}
    if proton is None:
        raise RuntimeError("triton.profiler (proton) is not available in this environment.")
    use_cuda_graph = effective_use_cuda_graph(use_cuda_graph)

    # Warmup outside Proton session
    for _ in range(max(0, warmup)):
        if flush_l2:
            _flush_l2_cache()
        f(*tuple_of_args, **kwargs)
    torch.cuda.synchronize()

    # Measurement session
    profile_base = _build_profile_base("tree", proton_output_dir, proton_file_label)
    session_id = proton.start(
        name=profile_base,
        context=proton_context,
        data="tree",
        backend=proton_backend,
    )
    try:
        runner = _prepare_runner(f, tuple_of_args, kwargs, use_cuda_graph=use_cuda_graph)
        torch.cuda.synchronize()
        for _ in range(max(1, repeat)):
            if flush_l2:
                _flush_l2_cache()
            with proton.scope(proton_scope_name):
                runner()
        torch.cuda.synchronize()
    finally:
        proton.finalize(session=session_id)

    # Parse hatchet
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

    if mean_ns <= 0:
        raise RuntimeError(
            f"Failed to extract GPU timing from Proton hatchet data: "
            f"scope '{proton_scope_name}' not found or reported zero time. "
            f"Profile file: {path}"
        )

    return {"mean": mean_ns / 1e6}
