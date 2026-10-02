"""TileBench++ Trn2 measurement on the native PyTorch Neuron stack.

The formal Trn2 comparison is PyTorch eager on ``torch.device("neuron")`` against the NKI
implementation called with the same native neuron tensors (nki 0.7 dispatches it through
torch_neuronx). No torch_xla, no ``xm.mark_step`` and no torch.compile are involved; the
legacy torch-xla orchestrator (nki_orchestrator.py, nki_profile_worker.py) is not used here.

Per case and backend (torch eager, NKI), in this process:

1. inputs moved to the neuron device; first call (compile) timed separately;
2. output verified against the CPU reference with the operator's verify tolerance; a torch
   eager output that fails it is a baseline failure, so no speedup is formed for the case;
3. ``warmup`` calls, then ``repeat`` synchronized calls timed on the host (``wall`` stats);
4. ``repeat`` more calls, each inside its own ``torch.profiler`` session with
   ``NeuronConfig(modes=[RUNTIME])``; the device time of one call is the busy sum of the
   ``nc_exec_running`` executions in that session (per-core copies of one execution merged by
   (model id, flow id)). The time is accepted only if every call ran at least one execution,
   every call ran the same multiset of executables, and the last profiled output verifies.

The reported latency (``mean``) is the mean device busy sum: the sum over the device executions
of one run() of each execution's duration (the per-core copies of one execution are merged
first), i.e. idle gaps between executions are not counted. That is the definition of the GPU path
(timer.report_benchmark: Proton sums the durations of the CUDA kernels inside the scope and
divides by repeat). Host wall time is recorded next to it and never substituted for it.

torch.compile is switched off while a backend is measured on the device (``measure`` runs inside
``torch_compile_disabled``): a ``torch.compile`` call inside an implementation runs its function
eagerly, so the PyTorch baseline is plain eager (an implementation that cannot run without
compile fails instead). The CPU reference is computed before, unchanged. Dynamo frame counters
are recorded per measurement as evidence.
"""
from __future__ import annotations

import collections
import contextlib
import importlib.metadata
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

NEURON_DEFAULT_WARMUP = 1
NEURON_DEFAULT_REPEAT = 3

# Operators whose native PyTorch eager baseline is unresolved: the NKI side is still verified
# and timed, but torch is not run and no speedup is formed.
BASELINE_UNRESOLVED = {
    "bitonic_sort": "native PyTorch eager baseline does not finish in reasonable time "
                    "(data-dependent shapes recompile per bitonic stage)",
    "radix_sort": "native PyTorch eager baseline (torch.sort) returns correct results but no "
                  "device execution was observed by the profiler; execution location unconfirmed",
    "block_sparse_attention": "the PyTorch baseline is flex_attention, which cannot run without "
                              "torch.compile (its eager path compiles too); no pure eager baseline",
    "flash_attention": "native PyTorch eager scaled_dot_product_attention (fp16) fails the operator's "
                       "verify tolerance (~10x the error of the CPU reference and of NKI)",
}

_ENV_DONE = False


@contextlib.contextmanager
def torch_compile_disabled():
    """Every torch.compile'd callable runs eagerly inside (the formal Trainium baseline is eager)."""
    import torch._dynamo

    before = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        yield
    finally:
        torch._dynamo.config.disable = before


def _dynamo_frames() -> int:
    try:
        from torch._dynamo.utils import counters
    except ImportError:
        return 0
    return int(sum(counters.get("frames", {}).values()))


def hardware_identity() -> dict | None:
    """What this Trainium host is, for the result namespace and its provenance; None when no
    Neuron device is present. Device configuration and software versions only (no instance id
    or host name)."""
    try:
        out = subprocess.run(["neuron-ls"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"instance-type:\s*(\S+)", out)
    if not m:
        return None
    instance = m.group(1)
    lnc = re.search(r"logical-neuroncore-config:\s*(\d+)", out)
    row = re.search(r"^\|\s*\d+\s*\|\s*(\d+)\s*\|\s*[\d-]+\s*\|\s*([\d.]+\s*\w+)\s*\|", out, re.M)
    versions = {}
    for pkg in ("torch", "torch-neuronx", "nki", "neuronx-cc"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    try:
        driver = Path("/sys/module/neuron/version").read_text().strip()
    except OSError:
        driver = None
    return {"label": instance.split(".")[0].upper(), "instance_type": instance,
            "logical_neuroncore_config": int(lnc.group(1)) if lnc else None,
            "neuron_cores_per_device": int(row.group(1)) if row else None,
            "device_memory": row.group(2) if row else None,
            "neuron_driver": driver, "packages": versions, "neuronx_cc": shutil.which("neuronx-cc"),
            "env": {k: os.environ[k] for k in ("NEURON_RT_VISIBLE_CORES", "NEURON_RT_NUM_CORES",
                                               "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS")
                    if k in os.environ}}


def _lnc_from_neuron_ls() -> str | None:
    try:
        out = subprocess.run(["neuron-ls"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"logical-neuroncore-config:\s*(\d+)", out)
    return m.group(1) if m else None


def prepare_environment(cache_root: Path) -> dict:
    """Launch settings of the native stack. Must run before torch_neuronx is imported.

    torch_neuronx spawns the ``neuronx-cc`` found on PATH and keeps NEFF / HLO / NKI-trace
    caches in global default directories; both are pinned so the run uses the compiler of this
    interpreter's environment and a cache inside the run's log namespace. Explicit user
    settings win (setdefault)."""
    global _ENV_DONE
    bindir = Path(sys.executable).parent
    if (bindir / "neuronx-cc").exists():
        os.environ["PATH"] = os.pathsep.join([str(bindir), os.environ.get("PATH", "")])
    tools = Path("/opt/aws/neuron/bin")
    if tools.is_dir() and str(tools) not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join([os.environ.get("PATH", ""), str(tools)])
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG") or _lnc_from_neuron_ls()
    if lnc:
        os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", lnc)
        os.environ.setdefault("NEURON_CC_FLAGS", f"--target trn2 --lnc {lnc}")
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    os.environ.setdefault("NEURON_RT_NUM_CORES", "1")
    cache_root = Path(cache_root)
    for var, sub in (("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"), ("TORCH_NEURONX_HLO_CACHE_DIR", "hlo"),
                     ("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "local"), ("NKI_TRACE_CACHE_URL", "nki_trace"),
                     ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TORCH_NEURONX_DEBUG_DIR", "debug")):
        os.environ.setdefault(var, str(cache_root / sub))
    _ENV_DONE = True
    return {"neuronx_cc": shutil.which("neuronx-cc"),
            "env": {k: os.environ[k] for k in sorted(os.environ)
                    if k.startswith(("NEURON_", "TORCH_NEURONX_", "NKI_TRACE"))}}


class NativeNeuron:
    """The neuron device of the native stack: device moves and synchronization."""

    def __init__(self):
        import torch
        import torch_neuronx  # noqa: F401  (registers the neuron device)

        self.device = torch.device("neuron")
        torch.empty(1, device=self.device)
        sync = None
        for mod_name, attr in (("torch.neuron", "synchronize"), ("torch_neuronx", "synchronize")):
            try:
                mod = sys.modules.get(mod_name) or __import__(mod_name, fromlist=[attr])
            except ImportError:
                continue
            if callable(getattr(mod, attr, None)):
                sync = getattr(mod, attr)
                break
        if sync is None:
            raise RuntimeError("no native neuron synchronize API (torch.neuron / torch_neuronx)")
        self.sync = sync

    def _map(self, obj, fn):
        import torch

        if isinstance(obj, torch.Tensor):
            return fn(obj)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._map(o, fn) for o in obj)
        return obj

    def to_device(self, obj):
        return self._map(obj, lambda t: t.to(self.device))

    def to_cpu(self, obj):
        return self._map(obj, lambda t: t.cpu())


def _same(a, b) -> bool:
    import torch

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _device_call(rt: NativeNeuron, fn, args, kw, profile_dir: Path, label: str) -> dict:
    """One synchronized call inside its own profiler session; its device executions."""
    from torch.profiler import ProfilerActivity, profile
    from torch_neuronx.profiling import NeuronConfig, ProfileMode

    from tilebench.core import nki_timer
    from tilebench.neuron_diag.intervals import aggregate

    profile_dir.mkdir(parents=True, exist_ok=False)
    cfg = NeuronConfig(modes=[ProfileMode.RUNTIME], profile_output_dir=str(profile_dir))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1], experimental_config=cfg):
        rt.sync()
        out = fn(*args, **kw)
        rt.sync()
    sess = nki_timer.find_session_dir(str(profile_dir))
    execs, _ = nki_timer.load_session_executions(sess, data_path=str(profile_dir / "ne"), display_name=label)
    agg = aggregate([{"start_ns": e["start_ns"], "end_ns": e["end_ns"],
                      "execution_id": (str(e.get("model_id")), tuple(e["flow_id"]))} for e in execs])
    agg["model_ids"] = sorted(str(e.get("model_id")) for e in execs)
    agg["out"] = out
    return agg


def measure(fn, kw: dict, inputs, ref_output, **opts) -> dict:
    """Verify and time one backend on one case, with torch.compile disabled. Returns
    {"ok", "unsupported", "err", "ms", "stats"}; ``ms`` is the mean device busy sum, nan when it
    is unavailable (``err`` says why)."""
    with torch_compile_disabled():
        return _measure(fn, kw, inputs, ref_output, **opts)


def _measure(fn, kw: dict, inputs, ref_output, *, rt: NativeNeuron, warmup: int, repeat: int,
             atol, rtol, profile_dir: Path, label: str) -> dict:
    from tilebench.core.verifier import verify

    res = {"ok": False, "unsupported": False, "err": "", "ms": float("nan"), "stats": {}}
    stats = res["stats"]
    stats.update(timing_method="native_device_trace", timing_definition="busy_sum", warmup=warmup,
                 repeat=repeat, fallback_status="unable_to_determine")
    import torch._dynamo

    stats["torch_compile_disabled"] = bool(torch._dynamo.config.disable)
    frames0 = _dynamo_frames()
    try:
        x = rt.to_device(inputs)
        rt.sync()
        t0 = time.perf_counter()
        out = fn(*x, **kw)
        rt.sync()
        stats["first_call_ms"] = (time.perf_counter() - t0) * 1e3
        out_cpu = rt.to_cpu(out)
    except Exception as e:  # noqa: BLE001 - reported per case, never a latency
        # NotImplementedError is how an implementation declares a dtype/shape unsupported
        res["unsupported"] = isinstance(e, NotImplementedError)
        res["err"] = f"first call raised: {type(e).__name__}: {e}"[:2000]
        return res
    ok, err = verify(out_cpu, ref_output, atol=atol, rtol=rtol)
    if not ok:
        res["err"] = f"verification failed: {err}"[:2000]
        return res
    res["ok"] = True
    mutates = not _same(rt.to_cpu(x), inputs)
    stats["run_mutates_inputs"] = mutates

    def args():
        nonlocal x
        if mutates:
            x = rt.to_device(inputs)
            rt.sync()
        return x

    try:
        for _ in range(warmup):
            a = args()
            fn(*a, **kw)
            rt.sync()
        wall, dispatch = [], []
        for _ in range(repeat):
            a = args()
            t0 = time.perf_counter()
            o = fn(*a, **kw)
            t1 = time.perf_counter()
            rt.sync()
            wall.append((time.perf_counter() - t0) * 1e3)
            dispatch.append((t1 - t0) * 1e3)
            del o
        stats.update(wall_ms_mean=statistics.fmean(wall), wall_samples_ms=wall,
                     dispatch_median_ms=statistics.median(dispatch))
        iters = []
        for i in range(repeat):
            iters.append(_device_call(rt, fn, args(), kw, profile_dir / f"iter{i}", f"{label}-{i}"))
    except Exception as e:  # noqa: BLE001
        res["err"] = f"timed loop raised: {type(e).__name__}: {e}"[:2000]
        return res
    ok_last, err_last = verify(rt.to_cpu(iters[-1]["out"]), ref_output, atol=atol, rtol=rtol)
    multisets = {tuple(sorted(collections.Counter(it["model_ids"]).items())) for it in iters}
    problems = []
    if not ok_last:
        problems.append(f"profiled output does not verify: {err_last[:300]}")
    if any(it["n_executions"] == 0 for it in iters):
        problems.append("a timed call recorded no device execution")
    if len(multisets) != 1:
        problems.append("executables differ between timed calls (compile or different NEFF in window)")
    stats["dynamo_frames_compiled"] = _dynamo_frames() - frames0
    stats.update(executions_per_call=[it["n_executions"] for it in iters],
                 artifact_identity=sorted({m for it in iters for m in it["model_ids"]}),
                 overlap_between_executions=[it["overlap_between_executions"] for it in iters],
                 profile_dir=str(profile_dir))
    if problems:
        stats["device_timing_status"] = "unavailable"
        res["err"] = "device timing unavailable: " + "; ".join(problems)
        return res
    busy = [it["busy_sum_ms"] for it in iters]
    stats.update(device_timing_status="available", device_samples_ms=busy, mean=statistics.fmean(busy),
                 union_ms=[it["union_ms"] for it in iters], span_ms=[it["span_ms"] for it in iters])
    res["ms"] = stats["mean"]
    return res


def speedup(torch_ms: float, nki_ms: float) -> float:
    """speedup_nki = torch_eager_ms / nki_ms; nan unless both latencies exist."""
    if not (torch_ms > 0 and nki_ms > 0) or math.isnan(torch_ms) or math.isnan(nki_ms):
        return float("nan")
    return torch_ms / nki_ms
