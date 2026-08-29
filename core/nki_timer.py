"""Neuron on-device profiling for an EXACT, already-identified NEFF.

Proton cannot observe NeuronDevices (its backends are CUPTI/RocTracer only), so
both the torch-on-Neuron baseline graph and the NKI kernel graph are timed with
AWS neuron-profile / neuron-explorer, which uses on-chip profiling hardware to
report true on-device latency — the Trainium analog of Proton/NCU on NVIDIA.

This module deliberately does NOT locate NEFFs. Historically it globbed several
directories and picked the newest-mtime ``*.neff``, which silently profiles the
wrong graph after autotune sweeps, compile-cache hits, or when stale dumps from
earlier runs share the directory. Artifact identity now lives in
``core/nki_artifact.py`` (exactly-one validated pair inside a private per-spec
root, ``AwsNeuronCustomNativeKernel`` marker checks, SHA256s) and the process
design in ``core/nki_orchestrator.py`` / ``core/nki_profile_worker.py``.
``profile_neff`` here only *times* the exact file it is handed:

  1. ``$NEURON_PROFILE_BIN capture`` re-executes that NEFF on the device and
     writes an NTFF execution trace (``--profile-nth-exec`` skips warmup
     executions).
  2. ``$NEURON_PROFILE_BIN view --output-format summary-json`` reads the trace;
     its ``total_time`` (seconds) is returned as ``mean`` in milliseconds.

DEBUG ENV (auto-set below): the Neuron compiler only dumps the NEFF + HLO pair
consumed by the identity layer when NEURON_FRAMEWORK_DEBUG=1;
XLA_IR_DEBUG/XLA_HLO_DEBUG add the HLO/IR info that source-correlates the
profile. The profile worker sets these before importing torch-xla; the
setdefault here is a backstop for ad-hoc callers.

NOTE on inputs: the installed neuron-explorer capture accepts optional named
``.npy`` ifmaps (``IN1 x.npy ...``) but we do not currently supply them, so the
replayed execution uses the tool's default input contents. This is recorded as
``profile_input_mode`` in the returned stats and in the per-spec manifest.

Binary: recent SDKs ship ``neuron-profile``; the newest deprecate it in favor
of ``neuron-explorer`` (same subcommands). Override with $NEURON_PROFILE_BIN.
"""
import glob
import json
import os
import subprocess

import torch

for _flag in ("NEURON_FRAMEWORK_DEBUG", "XLA_IR_DEBUG", "XLA_HLO_DEBUG"):
    os.environ.setdefault(_flag, "1")

NEURON_PROFILE_BIN = os.environ.get("NEURON_PROFILE_BIN", "neuron-profile")

PROFILE_INPUT_MODE = "tool_default"  # capture run without explicit ifmaps


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


def _run_cli(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed (rc={r.returncode}): {r.stderr.strip()[:300]}")
    return r.stdout


def _total_time_ms(summary_json_text: str) -> float:
    """Extract on-device total_time (seconds) from `view` summary-json → ms.

    The summary may be a dict keyed by NeuronCore, a flat dict, or a list of
    rows; take the max total_time across rows (verified schema on trn2,
    neuron-explorer 2.32).
    """
    data = json.loads(summary_json_text)

    if isinstance(data, dict) and not any(k in data for k in ["total_time", "summary", "rows"]):
        rows = [v for v in data.values() if isinstance(v, dict)]
    else:
        rows = data if isinstance(data, list) else data.get("summary", data.get("rows", [data]))
        if isinstance(rows, dict):
            rows = [rows]

    times = [float(r["total_time"]) for r in rows if isinstance(r, dict) and "total_time" in r]
    if not times:
        raise RuntimeError(f"no 'total_time' in neuron-profile summary-json: {summary_json_text[:300]}")
    return max(times) * 1e3  # seconds -> ms


def profile_neff(neff_path: str, *, warmup: int = 10, out_dir: str,
                 tag: str = "profile") -> dict:
    """Hardware-time ONE exact NEFF via capture/view; artifacts kept in out_dir.

    Returns a stats dict whose ``mean`` (ms) matches what
    ``core.timer.report_benchmark`` returns for the GPU backends.
    """
    if not os.path.isfile(neff_path):
        raise FileNotFoundError(f"profile_neff: no such NEFF: {neff_path!r}")
    os.makedirs(out_dir, exist_ok=True)

    nth = max(2, int(warmup) + 1)  # profile a warm execution
    ntff_stem = os.path.join(out_dir, tag)
    _run_cli([
        NEURON_PROFILE_BIN, "capture",
        "-n", neff_path, "-s", f"{ntff_stem}.ntff",
        f"--profile-nth-exec={nth}",
    ])
    ntff = f"{ntff_stem}_exec_{nth}.ntff"
    if not os.path.exists(ntff):  # naming fallback within OUR private out_dir
        found = sorted(glob.glob(os.path.join(out_dir, f"{tag}*.ntff")))
        if not found:
            raise RuntimeError("neuron-profile capture produced no .ntff")
        ntff = found[-1]
    summary = _run_cli([
        NEURON_PROFILE_BIN, "view",
        "--output-format", "summary-json",
        "-n", neff_path, "-s", ntff,
    ])
    summary_path = os.path.join(out_dir, f"{tag}_summary.json")
    with open(summary_path, "w") as f:
        f.write(summary)

    mean_ms = _total_time_ms(summary)
    return {
        "mean": mean_ms,
        "total_ms": mean_ms,
        "repeat": 1,          # neuron-profile reports a single hardware-timed exec
        "method": "neuron_profile",
        "neff": neff_path,
        "ntff": ntff,
        "summary_path": summary_path,
        "profile_nth_exec": nth,
        "profile_input_mode": PROFILE_INPUT_MODE,
    }
