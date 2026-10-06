"""Environment capture for a calibration run. Every field that cannot be
read is recorded as the string "unavailable" (never guessed)."""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import shutil
import socket
import subprocess
import time

UNAVAILABLE = "unavailable"

NVIDIA_FIELDS = ["name", "uuid", "pci.bus_id", "driver_version", "vbios_version", "pstate", "temperature.gpu",
                 "temperature.memory", "power.draw", "power.limit", "enforced.power.limit", "clocks.sm", "clocks.mem",
                 "clocks.max.sm", "clocks.max.mem", "clocks.applications.graphics", "clocks.applications.memory",
                 "clocks_event_reasons.active", "mig.mode.current", "persistence_mode", "compute_mode",
                 "ecc.mode.current", "utilization.gpu", "memory.used", "memory.total"]


def _run(cmd: list[str], timeout: float = 30.0) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def nvidia_query(index: int) -> dict:
    if not shutil.which("nvidia-smi"):
        return {f: UNAVAILABLE for f in NVIDIA_FIELDS}
    out: dict = {}
    for f in NVIDIA_FIELDS:   # one field at a time: an unknown field must not lose the others
        rc, txt = _run(["nvidia-smi", "-i", str(index), f"--query-gpu={f}", "--format=csv,noheader"])
        val = txt.strip().splitlines()[-1].strip() if txt.strip() else ""
        out[f] = val if rc == 0 and val and "not a valid field" not in txt.lower() else UNAVAILABLE
    return out


def nvidia_compute_apps() -> list[str] | str:
    if not shutil.which("nvidia-smi"):
        return UNAVAILABLE
    rc, txt = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_uuid", "--format=csv,noheader"])
    if rc != 0:
        return UNAVAILABLE
    return [line.strip() for line in txt.splitlines() if line.strip()]


def rocm_query() -> dict:
    if not shutil.which("rocm-smi"):
        return {"rocm-smi": UNAVAILABLE}
    rc, txt = _run(["rocm-smi", "--showproductname", "--showuniqueid", "--showtemp", "--showclocks", "--showpower",
                    "--showperflevel", "--showmemuse", "--json"])
    return {"rocm-smi": txt if rc == 0 else UNAVAILABLE}


def device_sample(backend: str, index: int) -> dict:
    """Clock/power/temperature snapshot taken before and after every mode."""
    t = time.time()
    if backend == "cuda":
        q = nvidia_query(index)
        keep = ("pstate", "temperature.gpu", "temperature.memory", "power.draw", "clocks.sm", "clocks.mem",
                "clocks_event_reasons.active", "utilization.gpu", "memory.used")
        return {"at": t, **{k: q[k] for k in keep}}
    if backend == "rocm":
        return {"at": t, **rocm_query()}
    return {"at": t, "note": UNAVAILABLE}


def precision_flags() -> dict:
    """Default precision controls. When torch has the fp32_precision API the
    legacy getters are NOT read (torch refuses matmuls after a mix of the two
    APIs); the probes set and record the new API explicitly per mode."""
    import torch
    m = torch.backends.cuda.matmul
    flags = {
        "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction": m.allow_fp16_reduced_precision_reduction,
        "torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction": m.allow_bf16_reduced_precision_reduction,
    }
    if hasattr(m, "fp32_precision"):
        flags["torch.backends.cuda.matmul.fp32_precision"] = m.fp32_precision
        flags["torch.backends.cudnn.fp32_precision"] = getattr(torch.backends.cudnn, "fp32_precision", UNAVAILABLE)
        flags["torch.backends.fp32_precision"] = getattr(torch.backends, "fp32_precision", UNAVAILABLE)
        flags["legacy getters"] = "not read (new API present)"
    else:
        flags["torch.backends.cuda.matmul.allow_tf32"] = m.allow_tf32
        flags["torch.get_float32_matmul_precision()"] = torch.get_float32_matmul_precision()
        flags["torch.backends.cudnn.allow_tf32"] = torch.backends.cudnn.allow_tf32
    try:
        flags["preferred_blas_library"] = str(torch.backends.cuda.preferred_blas_library())
    except Exception:  # noqa: BLE001
        flags["preferred_blas_library"] = UNAVAILABLE
    for env in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG",
                "TORCH_BLAS_PREFER_HIPBLASLT", "HIPBLASLT_ALLOW_TF32", "ROCBLAS_LAYER"):
        flags[f"env:{env}"] = os.environ.get(env, "<unset>")
    return flags


_BLAS_CHECK = r"""
import json, os, sys
res = {"checks": {}}
try:
    import torch
    hip = bool(getattr(torch.version, "hip", None))
    def run(name, fn):
        try:
            fn(); torch.cuda.synchronize(); res["checks"][name] = "ok"
        except Exception as e:
            res["checks"][name] = f"{type(e).__name__}: {str(e)[:240]}"
    a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    run("fp16 matmul", lambda: torch.matmul(a, a))
    x = torch.randint(-8, 8, (64, 64), dtype=torch.int8, device="cuda")
    run("int8 _int_mm", lambda: torch._int_mm(x, x))
    f8 = getattr(torch, "float8_e4m3fnuz" if hip else "float8_e4m3fn")
    f = torch.randn(64, 64, device="cuda").to(f8); one = torch.ones((), device="cuda")
    run("fp8 _scaled_mm", lambda: torch._scaled_mm(f, f.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
except Exception as e:
    res["checks"]["import"] = f"{type(e).__name__}: {e}"
maps = open(f"/proc/{os.getpid()}/maps").read()
libs = sorted({l.split()[-1] for l in maps.splitlines() if any(k in l for k in ("cublas", "hipblas", "rocblas"))})
res["libraries"] = libs
res["blas_dirs"] = sorted({os.path.dirname(p) for p in libs if os.path.basename(p).startswith(("libcublas", "libhipblas"))})
print(json.dumps(res))
"""


def blas_stack_check(env: dict | None = None, timeout: float = 300.0) -> dict:
    """Run tiny BLAS calls (fp16 GEMM, int8 and fp8 cuBLASLt/hipBLASLt GEMMs)
    in a child process with `env` and report which BLAS libraries were
    loaded. ok=False when a call fails or libcublas and libcublasLt come from
    different directories (e.g. LD_LIBRARY_PATH pulling a system cuBLASLt of
    another version under the wheel's cuBLAS)."""
    import json
    import sys
    try:
        r = subprocess.run([sys.executable, "-c", _BLAS_CHECK], capture_output=True, text=True, timeout=timeout,
                           env=env if env is not None else dict(os.environ))
        line = [l for l in r.stdout.splitlines() if l.startswith("{")][-1]
        res = json.loads(line)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    bad = {k: v for k, v in res.get("checks", {}).items() if v != "ok"}
    mixed = len(res.get("blas_dirs", [])) > 1
    res["ok"] = not bad and not mixed
    if mixed:
        res["error"] = f"BLAS libraries loaded from different directories: {res['blas_dirs']}"
    elif bad:
        res["error"] = f"BLAS calls failed: {bad}"
    res["LD_LIBRARY_PATH"] = (env if env is not None else os.environ).get("LD_LIBRARY_PATH", "<unset>")
    return res


def _pkg_versions() -> dict:
    out = {}
    wanted = ("torch", "triton", "nvidia-cublas", "nvidia-cuda-runtime", "nvidia-cudnn", "pytorch-triton-rocm",
              "cuda-python", "numpy")
    for d in md.distributions():
        name = (d.metadata["Name"] or "").lower()
        if name in wanted or name.startswith(("nvidia-cublas", "nvidia-cuda-runtime", "hipblaslt", "rocm")):
            out[d.metadata["Name"]] = d.version
    return out


def capture(backend: str, index: int) -> dict:
    import torch
    env: dict = {
        "captured_at": time.time(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "backend": backend,
        "packages": _pkg_versions(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda, "hip": getattr(torch.version, "hip", None)},
        "visible_devices": {k: os.environ.get(k, "<unset>") for k in
                            ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                             "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG")},
        "precision_flags_default": precision_flags(),
    }
    try:
        p = torch.cuda.get_device_properties(index)
        env["device_properties"] = {"name": p.name, "multi_processor_count": p.multi_processor_count,
                                    "total_memory_bytes": p.total_memory,
                                    "l2_cache_bytes": getattr(p, "L2_cache_size", UNAVAILABLE),
                                    "compute_capability": f"{p.major}.{p.minor}",
                                    "uuid": str(getattr(p, "uuid", UNAVAILABLE)),
                                    "gcn_arch_name": getattr(p, "gcnArchName", UNAVAILABLE)}
    except Exception as e:  # noqa: BLE001
        env["device_properties"] = {"error": str(e)}
    if backend == "cuda":
        env["nvidia_smi"] = nvidia_query(index)
        env["other_compute_processes"] = nvidia_compute_apps()
    elif backend == "rocm":
        env["rocm"] = rocm_query()
    try:
        import triton
        env["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        env["triton"] = UNAVAILABLE
    return env
