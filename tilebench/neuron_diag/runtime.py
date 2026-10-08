"""Runtime adapters: backend initialization, device transfer, synchronization,
baseline compilation and CPU-fallback evidence, one class per software stack.

The XLA adapter uses the mechanisms the existing benchmark already relies on
(``mark_step`` + ``wait_device_ops``; torch-xla's ``aten::*`` counters, which
torch-xla increments for every op it executes on the CPU instead of the device).

The native adapter never assumes an API: every entry point is *probed* on the
installed build and the one actually used is recorded. When no candidate exists
the adapter raises ``RuntimeUnavailable`` instead of substituting something that
only looks like it works (a sync that does not sync would make every wall-clock
number meaningless).
"""
from __future__ import annotations

import importlib
from typing import Any, Callable


class RuntimeUnavailable(RuntimeError):
    """The requested stack, or a required API of it, is not available here."""


def _map(obj, fn):
    import torch

    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(_map(v, fn) for v in obj)
    if isinstance(obj, list):
        return [_map(v, fn) for v in obj]
    return obj


# aten:: counters that torch-xla bumps for a device->host scalar read (.item(),
# Python int()/bool() of a tensor). They mark a host round trip inside run(), not an
# operator computed on the CPU, and are reported separately.
HOST_READ_COUNTERS = ("aten::_local_scalar_dense",)


class XlaRuntime:
    stack = "xla"

    def __init__(self):
        try:
            import torch_xla  # noqa: F401
            import torch_xla.core.xla_model as xm
            import torch_xla.debug.metrics as met
        except ImportError as e:
            raise RuntimeUnavailable(f"torch_xla is not importable: {e}") from e
        self._xm, self._met = xm, met
        self.api = {"sync": "xm.mark_step + xm.wait_device_ops",
                    "fallback": "torch_xla.debug.metrics aten::* counters"}
        self._fb_before: dict = {}

    def device(self):
        return self._xm.xla_device()

    def to_device(self, obj):
        dev = self.device()
        return _map(obj, lambda t: t.to(dev))

    def sync(self, _out=None) -> None:
        self._xm.mark_step()
        self._xm.wait_device_ops()

    def to_cpu(self, obj):
        return _map(obj, lambda t: t.cpu())

    def compile(self, fn: Callable) -> Callable:
        raise RuntimeUnavailable("torch.compile is not a mode of the XLA stack in this diagnostic")

    def _aten_counters(self) -> dict:
        return {n: self._met.counter_value(n) for n in self._met.counter_names()
                if n.startswith("aten::")}

    def fallback_begin(self) -> None:
        self._fb_before = self._aten_counters()

    def fallback_end(self) -> tuple[str, str]:
        after = self._aten_counters()
        grew = {n: v - self._fb_before.get(n, 0) for n, v in after.items()
                if v > self._fb_before.get(n, 0)}
        host_reads = {n: v for n, v in grew.items() if n in HOST_READ_COUNTERS}
        compute = {n: v for n, v in grew.items() if n not in HOST_READ_COUNTERS}
        note = (f"; host round trips (device->host scalar reads, not CPU compute): {host_reads}"
                if host_reads else "")
        if compute:
            return "confirmed_cpu_fallback", f"torch_xla aten:: CPU-fallback counters grew: {compute}{note}"
        if host_reads:
            return "host_scalar_read", ("torch_xla aten:: compute counters unchanged over the timed loop"
                                        + note)
        return ("no_fallback_observed_with_evidence",
                "torch_xla aten:: CPU-fallback counters unchanged over the timed loop")

    def compile_count(self) -> int:
        d = self._met.metric_data("CompileTime")
        return int(d[0]) if d else 0

    def compile_info(self) -> dict:
        return {}


# Candidate entry points of the native build, in probe order. They come from the
# torch-neuronx native examples; which one exists is decided on the installed
# build, and nothing is used without being found.
_NATIVE_SYNC_CANDIDATES = (("torch.neuron", "synchronize"), ("torch_neuronx", "synchronize"))
_NATIVE_DYNAMO_METRICS = (("torch_neuronx", "get_dynamo_metrics"),)


def _resolve(candidates) -> tuple[str, Callable] | None:
    import torch

    for mod_name, attr in candidates:
        try:
            if mod_name == "torch.neuron":
                mod = getattr(torch, "neuron", None)
            else:
                mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        fn = getattr(mod, attr, None) if mod is not None else None
        if callable(fn):
            return f"{mod_name}.{attr}", fn
    return None


class NativeRuntime:
    stack = "native"
    device_name = "neuron"

    def __init__(self):
        import torch

        try:
            import torch_neuronx  # noqa: F401  (registers the device on native builds)
        except ImportError as e:
            raise RuntimeUnavailable(f"torch_neuronx is not importable: {e}") from e
        try:
            torch.empty(1, device=self.device_name)
        except (RuntimeError, TypeError) as e:
            raise RuntimeUnavailable(
                f"torch has no '{self.device_name}' device in this build "
                f"(torch {torch.__version__}): {str(e)[:160]}") from e
        found = _resolve(_NATIVE_SYNC_CANDIDATES)
        if found is None:
            raise RuntimeUnavailable(
                "no device synchronize API found among "
                f"{['.'.join(c) for c in _NATIVE_SYNC_CANDIDATES]}; wall-clock timing would be "
                "meaningless without one")
        self._sync_name, self._sync = found
        metrics = _resolve(_NATIVE_DYNAMO_METRICS)
        self._metrics_name, self._metrics = metrics if metrics else (None, None)
        self.api = {"sync": self._sync_name, "dynamo_metrics": self._metrics_name,
                    "fallback": None}
        self._dynamo_before: dict = {}

    def device(self):
        import torch

        return torch.device(self.device_name)

    def to_device(self, obj):
        dev = self.device()
        return _map(obj, lambda t: t.to(dev))

    def sync(self, _out=None) -> None:
        self._sync()

    def to_cpu(self, obj):
        return _map(obj, lambda t: t.cpu())

    def compile(self, fn: Callable) -> Callable:
        import torch

        self._dynamo_before = _dynamo_counters()
        return torch.compile(fn, backend="neuron", dynamic=False)

    def fallback_begin(self) -> None:
        pass

    def fallback_end(self) -> tuple[str, str]:
        return ("unable_to_determine",
                "no CPU-fallback reporting API has been verified for this native build; "
                "outputs on the neuron device and a compiled callable are not evidence")

    def compile_count(self) -> int | None:
        c = _dynamo_counters().get("stats", {})
        return c.get("unique_graphs")

    def compile_info(self) -> dict:
        info: dict[str, Any] = {"dynamo_counters_delta": _counter_delta(self._dynamo_before,
                                                                         _dynamo_counters())}
        if self._metrics is not None:
            try:
                info["neuron_dynamo_metrics"] = repr(self._metrics())[:4000]
            except Exception as e:  # noqa: BLE001 - recorded, not interpreted
                info["neuron_dynamo_metrics_error"] = f"{type(e).__name__}: {e}"[:400]
        return info


def _dynamo_counters() -> dict:
    try:
        from torch._dynamo.utils import counters
    except ImportError:
        return {}
    return {k: dict(v) for k, v in counters.items()}


def _counter_delta(before: dict, after: dict) -> dict:
    out = {}
    for group, vals in after.items():
        prev = before.get(group, {})
        d = {str(k): v - prev.get(k, 0) for k, v in vals.items() if v != prev.get(k, 0)}
        if d:
            out[group] = d
    return out


def get_runtime(stack: str):
    if stack == "xla":
        return XlaRuntime()
    if stack == "native":
        return NativeRuntime()
    raise ValueError(f"unknown stack {stack!r}")
