"""Timing helpers for the NKI (AWS Neuron / Trainium) backend.

Proton cannot observe NeuronDevices, so the NKI backend is timed with an
XLA-synchronised wall-clock loop around ``impl_nki.run()``:

1. ``warmup`` calls, each closed by ``xm.mark_step()`` (compiles and caches
   the single-call graph now, so compilation stays out of the timed region),
   then one ``wait_device_ops()`` to drain.
2. ``repeat`` calls, each closed by ``xm.mark_step()`` **while its output is
   still referenced**, then one final ``wait_device_ops()``;
   mean latency = elapsed / repeat.

Stepping every iteration is load-bearing: XLA is lazy and prunes graphs with
no live output tensor, so discarding each ``run()`` result before a single
end-of-loop barrier would skip most of the kernel work and yield invalid
near-zero timings (PR #102 review). Marking a step each iteration while the
output is alive forces every repeat to be dispatched; ``mark_step`` is async,
so the loop still pipelines and the final ``wait_device_ops`` drains the tail.

The result is throughput-derived end-to-end latency, which may include
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

    # Warmup: step each iteration while the output is live so the per-call graph
    # is compiled, cached, and actually executed (not pruned).
    for _ in range(max(1, warmup)):
        out = fn(*inputs, **kwargs)
        xm.mark_step()
        del out
    xm.wait_device_ops()

    # Timed: step each iteration with its output still referenced, so every
    # repeat is dispatched. mark_step is async (the loop pipelines); the final
    # wait_device_ops drains the queue. See module docstring for why a single
    # end-of-loop barrier would under-measure.
    start = time.perf_counter()
    for _ in range(max(1, repeat)):
        out = fn(*inputs, **kwargs)
        xm.mark_step()
        del out
    xm.wait_device_ops()
    elapsed_ms = (time.perf_counter() - start) * 1e3

    n = max(1, repeat)
    return {
        "mean": elapsed_ms / n,
        "total_ms": elapsed_ms,
        "repeat": n,
        "method": "xla_wallclock",
    }
