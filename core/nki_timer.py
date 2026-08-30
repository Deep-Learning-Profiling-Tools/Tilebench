"""Neuron on-device timing from the runtime's own execution trace.

Proton cannot observe NeuronDevices (its backends are CUPTI/RocTracer only).
On Trainium the torch-on-Neuron baseline and the NKI kernel graph are timed
with the Neuron runtime's inspect facility (``NEURON_RT_INSPECT_*``, the same
mechanism ``neuron-explorer inspect`` uses):

* the profile worker (core/nki_profile_worker.py) runs the operator's
  ``run()`` ``warmup`` + ``repeat`` times on the case's REAL inputs and records
  a wall-clock window ``[t0, t1]`` (``time.time_ns()``) around every timed
  iteration;
* the runtime (``RUNTIME_INSPECT_ENV`` in core/nki_profile_spec.py) writes,
  per executed NEFF, the exact binary it loaded
  (``neff_<model_id>_vnc_0.neff``), a device trace of its first execution
  (``<model_id>_vnc_0.ntff``, for deep-dive analysis) and a system trace
  (``ntrace.pb``) with one ``nc_exec_running`` hardware event per execution;
* after the worker exits, the parent ingests the session with
  ``neuron-explorer view -d`` (parquet), sums the device time of every
  execution inside each window, and maps every executed ``model_id`` to one of
  this run's compiler-dumped NEFF/HLO pairs by SHA256 (the runtime-written
  NEFF is byte-identical to the compiler output) — so multi-graph operators
  (e.g. radix_sort's per-pass graphs) are timed as the sum of their graphs,
  and every timed byte is attributed to a validated artifact.

Nothing here locates artifacts by mtime, sequence number, or glob order; see
core/nki_artifact.py for the identity rules.

``profile_neff`` (``neuron-explorer capture``/``view`` of ONE NEFF, executed
with the tool's default input contents) remains only for the explicit
``$NKI_NEFF_PATH`` override, whose latency is reported as unverified.

Binary: ``neuron-explorer`` (override with $NEURON_PROFILE_BIN).
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import subprocess

import torch

NEURON_PROFILE_BIN = os.environ.get("NEURON_PROFILE_BIN", "neuron-explorer")

PROFILE_INPUT_MODE_INSPECT = "runtime_inspect_real_inputs"
PROFILE_INPUT_MODE_CAPTURE = "tool_default"  # capture run without explicit ifmaps

HW_EXEC_EVENT = "nc_exec_running"  # one per execution in the system trace


class NkiTraceError(RuntimeError):
    """The runtime trace does not support an unambiguous latency."""


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


# --------------------------------------------------------------------------
# runtime-inspect session: locate, ingest, load
# --------------------------------------------------------------------------

def find_session_dir(inspect_dir: str) -> str:
    """The EXACTLY ONE runtime session (``<inspect_dir>/i-*_pid_*/<ts>/ntrace.pb``)
    written by the profile worker; zero or several is an error."""
    traces = sorted(glob.glob(os.path.join(inspect_dir, "i-*_pid_*", "*", "ntrace.pb")))
    if len(traces) != 1:
        raise NkiTraceError(
            f"expected exactly one runtime-inspect session under {inspect_dir!r}, "
            f"found {len(traces)}: {traces} — was the worker launched with "
            f"NEURON_RT_INSPECT_SYSTEM_PROFILE=1?")
    return os.path.dirname(traces[0])


def session_neffs(session_dir: str) -> dict[str, str]:
    """model_id -> path of the NEFF the runtime executed (written by the runtime
    itself with NEURON_RT_INSPECT_DEVICE_PROFILE=1)."""
    out: dict[str, str] = {}
    for p in sorted(glob.glob(os.path.join(session_dir, "neff_*_vnc_*.neff"))):
        base = os.path.basename(p)[len("neff_"):-len(".neff")]
        model_id, _vnc = base.rsplit("_vnc_", 1)
        if model_id in out:
            raise NkiTraceError(f"several NEFF files for model {model_id} in {session_dir}: "
                                f"{out[model_id]}, {p}")
        out[model_id] = p
    return out


def ingest_session(session_dir: str, *, data_path: str, display_name: str) -> str:
    """Ingest a runtime session into parquet; returns the session's parquet dir.

    Only the system trace is needed for timing, so the per-NEFF device traces
    are not parsed here (3x faster ingestion; identical execution table). The
    NTFFs stay in the session dir — ``neuron-explorer view -d <session_dir>``
    ingests them fully for a deep-dive.
    """
    _run_cli([NEURON_PROFILE_BIN, "view", "-d", session_dir, "--ingest-only",
              "--data-path", data_path, "--display-name", display_name,
              "--ignore-instruction-trace", "--ignore-dma-trace", "--ignore-event-trace"])
    parquet_dir = os.path.join(data_path, "profiles", "global", f"{display_name}@latest")
    if not os.path.isfile(os.path.join(parquet_dir, "SystemProfileEvents.parquet")):
        raise NkiTraceError(f"ingestion of {session_dir} produced no "
                            f"SystemProfileEvents.parquet under {parquet_dir}")
    return parquet_dir


def load_executions(parquet_dir: str) -> list[dict]:
    """Every hardware execution in the session: one record per ``flow_id``
    (LNC>1 emits one ``nc_exec_running`` row per physical core; they are merged
    into ``[min start, max end]``)."""
    import pyarrow.parquet as pq  # Neuron-host dependency (see requirements.txt)

    table = pq.read_table(
        os.path.join(parquet_dir, "SystemProfileEvents.parquet"),
        columns=["name", "start_ts", "end_ts", "flow_id", "pcore_idx",
                 "extra_attributes_json"])
    by_flow: dict[tuple, dict] = {}
    for row in table.to_pylist():
        if row["name"] != HW_EXEC_EVENT:
            continue
        flow = tuple(row["flow_id"] or ())
        if not flow:
            raise NkiTraceError(f"{HW_EXEC_EVENT} event without flow_id: {row}")
        attrs = json.loads(row["extra_attributes_json"] or "{}")
        model_id = str(attrs.get("model_id", ""))
        if not model_id:
            raise NkiTraceError(f"{HW_EXEC_EVENT} event without model_id: {row}")
        rec = by_flow.setdefault(flow, {"flow_id": list(flow), "model_id": model_id,
                                        "start_ns": row["start_ts"], "end_ns": row["end_ts"],
                                        "pcores": 0})
        if rec["model_id"] != model_id:
            raise NkiTraceError(f"flow {flow} mixes models {rec['model_id']} and {model_id}")
        rec["start_ns"] = min(rec["start_ns"], row["start_ts"])
        rec["end_ns"] = max(rec["end_ns"], row["end_ts"])
        rec["pcores"] += 1
    return sorted(by_flow.values(), key=lambda r: (r["start_ns"], r["end_ns"]))


def load_session_executions(session_dir: str, *, data_path: str,
                            display_name: str) -> tuple[list[dict], str]:
    """Default executions loader used by the orchestrator (ingest + load)."""
    parquet_dir = ingest_session(session_dir, data_path=data_path, display_name=display_name)
    return load_executions(parquet_dir), parquet_dir


# --------------------------------------------------------------------------
# window timing (pure; unit-tested without hardware)
# --------------------------------------------------------------------------

def time_windows(executions: list[dict], windows: list, *, tag: str = "") -> dict:
    """Device time of the executions inside each ``[t0_ns, t1_ns]`` window.

    Every window is one timed ``run()`` iteration; its latency is the sum of
    the device time of the executions it contains (several for multi-graph
    operators). Fail-loud checks: every window holds >= 1 execution, no
    execution straddles a window boundary, and every window holds the SAME
    multiset of model_ids (a differing pattern means dropped trace events or a
    graph structure that changes between calls — either way no single number
    describes the operator).
    """
    if not windows:
        raise NkiTraceError(f"[{tag}] no timed windows recorded")
    per_iter_ms: list[float] = []
    patterns: list[tuple] = []
    per_model: dict[str, list[float]] = {}
    for i, (t0, t1) in enumerate(windows):
        inside = [e for e in executions if t0 <= e["start_ns"] and e["end_ns"] <= t1]
        straddling = [e for e in executions
                      if (e["start_ns"] < t0 < e["end_ns"]) or (e["start_ns"] < t1 < e["end_ns"])]
        if straddling:
            raise NkiTraceError(
                f"[{tag}] window {i} [{t0}, {t1}] cuts through execution(s) "
                f"{[e['model_id'] for e in straddling]} — clock mismatch between the worker and the trace")
        if not inside:
            raise NkiTraceError(
                f"[{tag}] window {i} [{t0}, {t1}] contains no {HW_EXEC_EVENT} execution "
                f"({len(executions)} executions in the session) — the timed run() executed "
                f"no graph, or the runtime trace is incomplete")
        patterns.append(tuple(sorted(e["model_id"] for e in inside)))
        per_iter_ms.append(sum(e["end_ns"] - e["start_ns"] for e in inside) / 1e6)
        for e in inside:
            per_model.setdefault(e["model_id"], []).append((e["end_ns"] - e["start_ns"]) / 1e6)
    if len(set(patterns)) != 1:
        raise NkiTraceError(
            f"[{tag}] the executed-graph pattern differs between timed iterations "
            f"({len(set(patterns))} distinct patterns, e.g. {patterns[0]} vs "
            f"{next(p for p in patterns if p != patterns[0])}) — trace events dropped "
            f"(raise NEURON_RT_INSPECT_SYS_TRACE_MAX_EVENTS_PER_NC) or a non-deterministic "
            f"graph structure; no single latency describes this operator")
    pattern = patterns[0]
    n = len(windows)
    return {
        "mean": statistics.fmean(per_iter_ms),
        "min": min(per_iter_ms),
        "max": max(per_iter_ms),
        "stdev": statistics.pstdev(per_iter_ms) if n > 1 else 0.0,
        "repeat": n,
        "per_iteration_ms": per_iter_ms,
        "executions_per_iteration": len(pattern),
        "per_model": {m: {"count_per_iteration": pattern.count(m),
                          "mean_ms": statistics.fmean(v)}
                      for m, v in per_model.items()},
        "method": "neuron_rt_inspect",
        "profile_input_mode": PROFILE_INPUT_MODE_INSPECT,
    }


# --------------------------------------------------------------------------
# explicit-override path only: capture/view of ONE NEFF (tool default inputs)
# --------------------------------------------------------------------------

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

    Used only for the explicit ``$NKI_NEFF_PATH`` override: the NEFF is
    re-executed by the tool with its default input contents, so the number is
    audit-only (never verified against the case).
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
        "repeat": 1,          # a single hardware-timed execution
        "method": "neuron_explorer_capture",
        "neff": neff_path,
        "ntff": ntff,
        "summary_path": summary_path,
        "profile_nth_exec": nth,
        "profile_input_mode": PROFILE_INPUT_MODE_CAPTURE,
    }
