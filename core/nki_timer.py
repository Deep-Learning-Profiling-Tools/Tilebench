"""Timing helpers for the NKI (AWS Neuron / Trainium) backend.

Proton cannot observe NeuronDevices (its backends are CUPTI/RocTracer only), so
the NKI backend is timed with **AWS neuron-profile**, which uses on-chip
profiling hardware to report true on-device kernel latency — the Trainium analog
of Proton/NCU on NVIDIA. This is now the ONLY NKI timing mode (the earlier XLA
wall-clock loop was removed: it included graph-dispatch / host overhead and was
not apples-to-apples with the Proton numbers used for the GPU backends).

Pipeline (per ``bench_nki`` call):
  1. Run ``impl_nki.run()`` once on the XLA device to compile the kernel and make
     the Neuron compiler emit the NEFF artifact.
  2. ``neuron-profile capture`` re-executes that NEFF on the device and writes an
     NTFF execution trace (``--profile-nth-exec`` skips warmup executions).
  3. ``neuron-profile view --output-format summary-json`` reads the trace and
     reports ``total_time`` (on-device execution time); we return it as ``mean``.

DEBUG ENV (auto-set below): the Neuron compiler only saves the NEFF (consumed in
step 2) when NEURON_FRAMEWORK_DEBUG=1; XLA_IR_DEBUG/XLA_HLO_DEBUG add the HLO/IR
info that source-correlates the profile. This module sets all three via
os.environ.setdefault at import — early enough because the engine imports it
BEFORE the first impl_nki.run() compiles the kernel — so callers normally only
need their platform selector, e.g.:

    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 PYTHONPATH=. \\
    python scripts/run_bench.py --operator <op> --tile-language nki

(setdefault preserves any value you export yourself; exporting at launch is the
most robust if torch_xla happens to get imported before this module.)

STATUS: written against the neuron-profile docs but NOT yet run on a trn1/trn2
box. The items most likely to need adjustment there are marked ``# VERIFY ON
TRN2``:
  - the NEFF location/name emitted by a torch_xla run,
  - the exact ``neuron-profile`` capture/view flags, the summary-json schema, and
    the ``total_time`` field + its units (docs say seconds),
  - whether the installed binary is ``neuron-profile`` or ``neuron-explorer``
    (override with the ``NEURON_PROFILE_BIN`` env var).

Refs: AWS Neuron docs — "Profile a NKI Kernel" and "Neuron Profile User Guide".
"""
import glob
import json
import os
import subprocess
import tempfile

import torch

# The Neuron compiler saves the NEFF (consumed by neuron-profile below) only when
# NEURON_FRAMEWORK_DEBUG is set; XLA_IR_DEBUG/XLA_HLO_DEBUG add HLO/IR debug info
# for source-correlated profiles. Set them here so callers don't need the
# launch-time env: these are read when the kernel is first COMPILED (the first
# impl_nki.run()), which the engine triggers AFTER importing this module, so
# import-time setdefault is early enough. setdefault preserves user-exported values.
for _flag in ("NEURON_FRAMEWORK_DEBUG", "XLA_IR_DEBUG", "XLA_HLO_DEBUG"):
    os.environ.setdefault(_flag, "1")

# Binary that exposes the `capture` / `view` subcommands. Recent SDKs ship it as
# `neuron-profile`; the newest docs also use `neuron-explorer`. Override via env.
NEURON_PROFILE_BIN = os.environ.get("NEURON_PROFILE_BIN", "neuron-profile")
# Optional explicit NEFF path; otherwise we auto-discover the freshest *.neff.
NEFF_PATH_ENV = "NKI_NEFF_PATH"
# Directories the Neuron compiler may drop the NEFF into (relative to CWD).  # VERIFY ON TRN2
_NEFF_SEARCH_DIRS = (".", "compiler_workdir", "./neuronxcc-*", os.environ.get("NEURON_CC_FLAGS_CACHE_DIR", ""))


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


def _find_neff() -> str:
    """Locate the NEFF for the kernel under test (newest *.neff in the search dirs).

    Honors $NKI_NEFF_PATH if set. The NEFF is written when the kernel is first
    COMPILED — in the engine flow that is the correctness run *preceding*
    bench_nki, not a write inside it (the timed call reuses the cached graph) —
    so we take the newest artifact rather than gating on a bench-local timestamp.
    Raises if none found (usually means the Neuron debug env was not active at
    compile time — but this module auto-sets it; see the module docstring).  # VERIFY ON TRN2
    """
    explicit = os.environ.get(NEFF_PATH_ENV)
    if explicit:
        if not os.path.exists(explicit):
            raise RuntimeError(f"{NEFF_PATH_ENV}={explicit!r} does not exist")
        return explicit
    candidates = []
    for d in _NEFF_SEARCH_DIRS:
        if not d:
            continue
        candidates += glob.glob(os.path.join(d, "**", "*.neff"), recursive=True)
        candidates += glob.glob(os.path.join(d, "*.neff"))
    if not candidates:
        raise RuntimeError(
            "No NEFF found. The compiler saves it only with NEURON_FRAMEWORK_DEBUG=1 "
            f"(auto-set by this module); if it still isn't emitted, set ${NEFF_PATH_ENV} "
            "to the NEFF path explicitly."
        )
    return max(set(candidates), key=os.path.getmtime)


def _run_cli(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed (rc={r.returncode}): {r.stderr.strip()[:300]}")
    return r.stdout


def _total_time_ms(summary_json_text: str) -> float:
    """Extract on-device total_time (docs: seconds) from `view` summary-json.  # VERIFY ON TRN2

    The summary may be a dict or a list of per-NeuronCore rows; take the max
    total_time across rows and convert seconds -> ms.
    """
    data = json.loads(summary_json_text)
    rows = data if isinstance(data, list) else data.get("summary", data.get("rows", [data]))
    if isinstance(rows, dict):
        rows = [rows]
    times = [float(r["total_time"]) for r in rows if isinstance(r, dict) and "total_time" in r]
    if not times:
        raise RuntimeError(f"no 'total_time' in neuron-profile summary-json: {summary_json_text[:300]}")
    return max(times) * 1e3  # seconds -> ms


def bench_xla_wallclock(fn, inputs, kwargs=None, *, warmup=10, repeat=100):
    """Host wall-clock timing of ``fn(*inputs)`` on the XLA (Neuron) device.

    Used for the torch-on-Neuron reference column (``torch_nki_ms``): native
    torch ops lower through torch_xla to a fused Neuron graph, which
    neuron-profile's NEFF auto-discovery cannot reliably isolate, so the
    reference is timed as a wall-clock median around ``mark_step`` /
    ``wait_device_ops`` instead. This INCLUDES graph-dispatch/host overhead —
    when validating on trn2, sanity-check comparability against the
    neuron-profile numbers ``bench_nki`` reports for the NKI kernels.
    # VERIFY ON TRN2
    """
    import time as _time

    xm = _xm()
    kwargs = kwargs or {}

    def _once() -> float:
        t0 = _time.perf_counter()
        out = fn(*inputs, **kwargs)
        xm.mark_step()
        xm.wait_device_ops()
        del out
        return (_time.perf_counter() - t0) * 1e3

    for _ in range(warmup):
        _once()
    samples = sorted(_once() for _ in range(repeat))
    median_ms = samples[len(samples) // 2]
    return {
        "mean": median_ms,
        "p10": samples[int(len(samples) * 0.10)],
        "p90": samples[int(len(samples) * 0.90)],
        "repeat": repeat,
        "method": "xla_wallclock",
    }


def bench_nki(fn, inputs, kwargs=None, *, warmup=10, repeat=100):
    """Time ``fn(*inputs, **kwargs)`` on the Neuron device via neuron-profile.

    Returns a stats dict whose ``"mean"`` key (ms) matches what
    ``core.timer.report_benchmark`` returns for the GPU backends.
    """
    xm = _xm()
    kwargs = kwargs or {}

    # 1) Run once so the kernel is compiled and the NEFF exists on disk. The
    #    engine's correctness run usually compiled it already; this is harmless
    #    if the graph is cached (no recompile), and self-contained if bench_nki
    #    is ever called without a prior run. Keep the output live across
    #    mark_step so the graph isn't pruned, then drain.
    out = fn(*inputs, **kwargs)
    xm.mark_step()
    del out
    xm.wait_device_ops()

    neff = _find_neff()

    with tempfile.TemporaryDirectory(prefix="nki_prof_") as td:
        nth = max(2, int(warmup) + 1)  # profile a warm execution
        ntff_stem = os.path.join(td, "profile")
        # 2) Capture an on-device execution trace of the NEFF.  # VERIFY ON TRN2
        _run_cli([
            NEURON_PROFILE_BIN, "capture",
            "-n", neff, "-s", f"{ntff_stem}.ntff",
            f"--profile-nth-exec={nth}",
        ])
        ntff = f"{ntff_stem}_exec_{nth}.ntff"
        if not os.path.exists(ntff):  # naming fallback
            found = glob.glob(os.path.join(td, "*.ntff"))
            if not found:
                raise RuntimeError("neuron-profile capture produced no .ntff")
            ntff = found[-1]
        # 3) Read the summary and pull out on-device total_time.  # VERIFY ON TRN2
        summary = _run_cli([
            NEURON_PROFILE_BIN, "view",
            "--output-format", "summary-json",
            "-n", neff, "-s", ntff,
        ])

    mean_ms = _total_time_ms(summary)
    return {
        "mean": mean_ms,
        "total_ms": mean_ms,
        "repeat": 1,          # neuron-profile reports a single hardware-timed exec
        "method": "neuron_profile",
        "neff": neff,
    }
