"""Shared setup for the NKI 0.7.0b1 smoke tests (imported first by every test).

- Sets the evaluator's environment (LNC=2 on trn2, one logical NeuronCore per
  process) BEFORE torch_neuronx is imported.
- Points every compiler/runtime cache at NKI_SMOKE_CACHE_ROOT (default: a
  directory under the system temp dir), so the tests never reuse a cache entry
  written by another process.
- Provides `run_case` (records pass/fail, error text, wall time) and `emit`
  (prints one JSON line prefixed with SMOKE_RESULT, parsed by run_smoke.py).

Only torch on the neuron device is used to move data and call the kernel;
references are computed on CPU.
"""

import json
import os
import sys
import tempfile
import time
import traceback

TEST_NAME = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# ---- evaluator environment (must precede `import torch_neuronx`) -------------
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 2"
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_NUM_CORES"] = "1"
_venv_bin = os.path.dirname(sys.executable)
os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

_CACHE_ROOT = os.environ.get("NKI_SMOKE_CACHE_ROOT") or os.path.join(
    tempfile.gettempdir(), "nki_ref_smoke_cache"
)
_CACHE = os.path.join(_CACHE_ROOT, TEST_NAME)
for _var, _sub in [
    ("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"),
    ("TORCH_NEURONX_HLO_CACHE_DIR", "hlo"),
    ("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "neff_local"),
    ("NKI_TRACE_CACHE_URL", "trace"),
    ("NKI_COMPILE_CACHE_URL", "compile"),
    ("TORCH_NEURONX_DEBUG_DIR", "debug"),
]:
    _p = os.path.join(_CACHE, _sub)
    os.makedirs(_p, exist_ok=True)
    os.environ[_var] = _p

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_neuronx  # noqa: E402,F401  (registers the "neuron" device)

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

DEV = torch.device("neuron")
_CASES = []
_T0 = time.time()


def sync():
    torch.neuron.synchronize()


def to_dev(t):
    return t.to(DEV)


def short(exc, limit=900):
    """One-paragraph error text: exception type + message (whitespace-collapsed)."""
    msg = " ".join(str(exc).split())
    if len(msg) > limit:
        msg = msg[:limit] + " ..."
    return f"{type(exc).__name__}: {msg}"


def run_case(name, api, checked, fn, expect_error=False):
    """Run one case. fn() returns a dict of observations (or raises).

    expect_error=False: pass if fn returns without raising and obs.get('ok', True).
    expect_error=True : pass if fn raises (the error text is recorded).
    """
    t = time.time()
    rec = {"case": name, "api": api, "checked": checked, "negative": expect_error}
    try:
        obs = fn() or {}
        rec["observed"] = obs
        ok = bool(obs.get("ok", True))
        if expect_error:
            rec["status"] = "fail"
            rec["note"] = "expected an error but the call succeeded"
        else:
            rec["status"] = "pass" if ok else "fail"
    except Exception as e:  # noqa: BLE001
        rec["error"] = short(e)
        if expect_error:
            rec["status"] = "pass"
        else:
            rec["status"] = "fail"
            rec["traceback_tail"] = traceback.format_exc().splitlines()[-6:]
    rec["seconds"] = round(time.time() - t, 2)
    _CASES.append(rec)
    print(f"[{TEST_NAME}] {name}: {rec['status']} ({rec['seconds']} s)"
          + (f" -- {rec.get('error', '')[:300]}" if "error" in rec else ""), flush=True)
    return rec


def allclose(a, b, rtol, atol):
    a = a.detach().to("cpu").float()
    b = b.detach().to("cpu").float()
    err = (a - b).abs().max().item()
    return bool(torch.allclose(a, b, rtol=rtol, atol=atol)), err


def emit(extra=None):
    out = {
        "test": TEST_NAME,
        "status": "pass" if _CASES and all(c["status"] == "pass" for c in _CASES) else "fail",
        "seconds": round(time.time() - _T0, 2),
        "cases": _CASES,
    }
    if extra:
        out.update(extra)
    print("SMOKE_RESULT " + json.dumps(out, default=str), flush=True)
