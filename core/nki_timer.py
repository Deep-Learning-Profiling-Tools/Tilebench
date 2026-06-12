"""Timing helpers for the NKI (AWS Neuron / Trainium) backend.

Proton cannot observe NeuronDevices, so the NKI backend is timed with an
XLA-synchronised wall-clock loop around ``impl_nki.run()``:

1. ``warmup`` calls, closed by one full device sync (absorbs Neuron
   compilation and XLA graph tracing).
2. ``repeat`` calls timed as one batch, closed by a single
   ``xm.mark_step()`` + ``xm.wait_device_ops()``;
   mean latency = elapsed / repeat.

Batching the timed calls under one sync amortises XLA dispatch overhead
that would otherwise dominate microsecond-scale kernels. The trade-off is
that the result is throughput-derived end-to-end latency, which may include
graph-level overhead that the Proton numbers for the GPU backends do not
contain. Pure device-kernel latency would require ``nki.benchmark``'s
``nc_latency`` — but that wraps the raw kernel with numpy inputs, and the
impl contract deliberately hides the kernel behind ``run()`` (which owns
input adaptation, e.g. reshapes). Revisit if per-kernel device latency
becomes a requirement.

NOTE: structurally validated only — the failure paths are exercised on GPU
machines, but the happy path needs a run on a trn1/trn2 instance before NKI
numbers are trusted.
"""
import time

import torch


def _xm():
    # Deferred import: GPU-only machines have no torch_xla installed.
    from torch_xla.core import xla_model as xm
    return xm


def to_xla_device(inputs):
    """Move generator-produced tensors to the XLA (Neuron) device; pass scalars through."""
    dev = _xm().xla_device()
    return tuple(x.to(dev) if isinstance(x, torch.Tensor) else x for x in inputs)


def to_cpu(out):
    """Bring a Tensor (or tuple/list of Tensors) to CPU for cross-device verification."""
    if isinstance(out, torch.Tensor):
        return out.cpu()
    if isinstance(out, (tuple, list)):
        return type(out)(to_cpu(o) for o in out)
    return out


def bench_nki(fn, inputs, kwargs=None, *, warmup=10, repeat=100):
    """Time ``fn(*inputs, **kwargs)`` on the XLA device.

    Returns a stats dict whose ``"mean"`` key (ms) matches what
    ``core.timer.report_benchmark`` returns for the GPU backends.
    """
    xm = _xm()
    kwargs = kwargs or {}

    for _ in range(max(1, warmup)):
        fn(*inputs, **kwargs)
    xm.mark_step()
    xm.wait_device_ops()

    start = time.perf_counter()
    for _ in range(max(1, repeat)):
        fn(*inputs, **kwargs)
    xm.mark_step()
    xm.wait_device_ops()
    elapsed_ms = (time.perf_counter() - start) * 1e3

    n = max(1, repeat)
    return {
        "mean": elapsed_ms / n,
        "total_ms": elapsed_ms,
        "repeat": n,
        "method": "xla_wallclock",
    }
