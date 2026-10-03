"""Replay one catalogue pair exactly: the operator inputs and the autotune winner.

Every profiler harness (NCU, ROCm Compute Profiler) and the kernel-count probe
replay the same (operator, dtype, backend) the same way, so the configuration
the probe counts launches for is the configuration that gets profiled:

  - resolve_params(params, dtype)   the generator kwargs, dtype as a torch.dtype
  - apply_winner(impl, cfg, td)     install the autotune winner where run() reads
                                    it: per-kernel `_DEFAULT_<PREFIX>_CONFIG`
                                    dicts, the per-dtype `_DEFAULT_CONFIGS`, or
                                    `_DEFAULT_CONFIG`

impl.run() is then called with its default arguments, which take the
`_DEFAULT_CONFIG` path (no autotuning), so the replayed kernels are the winner's.
"""
import re
from types import SimpleNamespace

import torch

DTYPE_MAP = {
    "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32,
    "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
    "int8": torch.int8, "int32": torch.int32, "int64": torch.int64,
    "fp8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
    "fp8_e5m2":   getattr(torch, "float8_e5m2", None),
}


_PREFIXED_CONFIG_RE = re.compile(r"^_DEFAULT_([A-Z0-9]+)_CONFIG$")


def resolve_params(params: dict, dtype: str | None) -> dict:
    """The generator kwargs of one case with its dtype as a torch.dtype.

    `dtype` (the catalogue dtype string) is injected when the params carry no
    dtype; a string dtype already in the params is converted. Returns a new dict."""
    params = dict(params)
    if dtype is not None and "dtype" not in params:
        td = DTYPE_MAP.get(dtype)
        if td is None:
            raise RuntimeError(f"unknown dtype {dtype}")
        params["dtype"] = td
    elif isinstance(params.get("dtype"), str):
        td = DTYPE_MAP.get(params["dtype"])
        if td is not None:
            params["dtype"] = td
    return params


def inject_prefixed(impl, cfg: dict) -> dict:
    """Route prefixed winner keys into the per-kernel config dicts run() reads.

    Operators with more than one tunable kernel keep their configs in
    per-kernel dicts named `_DEFAULT_<PREFIX>_CONFIG` (linear_self_attention:
    `_DEFAULT_KV_CONFIG` / `_DEFAULT_OUT_CONFIG`), and the catalogue prefixes
    their winner keys to match (`kv_BLOCK_M`, `out_num_warps`, ...). Without
    this routing the override lands on an unused `_DEFAULT_CONFIG` attribute
    and the profiler silently replays the DEFAULT config instead of the winner.

    Mutates the target dicts in place (never rebinds them, so a `run()` that
    captured a reference still sees the update). Returns the unconsumed keys
    for the caller's existing single-config handling.
    """
    targets = {}
    for attr in dir(impl):
        m = _PREFIXED_CONFIG_RE.match(attr)
        if m:
            targets[m.group(1).lower()] = (attr, getattr(impl, attr))
    if not targets:
        return cfg

    leftover = {}
    for key, val in cfg.items():
        placed = False
        for pref, (attr, target) in targets.items():
            if not key.lower().startswith(pref + "_"):
                continue
            base = key[len(pref) + 1:]
            if isinstance(target, dict) and base in target:
                target[base] = val
                placed = True
            elif isinstance(target, SimpleNamespace) and hasattr(target, base):
                setattr(target, base, val)
                placed = True
            if placed:
                print(f"  cfg: {attr}[{base}] = {val}")
                break
        if not placed:
            leftover[key] = val
    return leftover


def apply_winner(impl, cfg: dict | None, torch_dtype=None) -> None:
    """Install the autotune winner `cfg` on the imported impl module.

    `torch_dtype` selects the `_DEFAULT_CONFIGS` entry of operators that keep
    one default config per dtype (matmul_fp32_fp16_fp8). No-op for an empty
    or missing winner, which replays the operator's own default config."""
    if not cfg:
        return
    cfg = inject_prefixed(impl, cfg)
    if not cfg:
        return
    existing = getattr(impl, "_DEFAULT_CONFIG", None)
    configs = getattr(impl, "_DEFAULT_CONFIGS", None)  # per-dtype dict, optional

    if existing is None and configs is not None:
        # Op uses per-dtype `_DEFAULT_CONFIGS[dtype]` (e.g. matmul_fp32_fp16_fp8).
        # Override the entry for the dtype we're about to run so the kernel
        # actually picks up `cfg`. Singular `_DEFAULT_CONFIG` is irrelevant
        # to this impl, so don't bother setting it.
        if torch_dtype in configs:
            cur = configs[torch_dtype]
            if isinstance(cur, dict):
                merged = dict(cur); merged.update(cfg)
                configs[torch_dtype] = merged
            else:
                merged = vars(cur).copy(); merged.update(cfg)
                configs[torch_dtype] = SimpleNamespace(**merged)
        else:
            # No entry for this dtype — create one wholesale.
            configs[torch_dtype] = SimpleNamespace(**cfg)
    elif isinstance(existing, dict):
        merged = dict(existing); merged.update(cfg)
        impl._DEFAULT_CONFIG = merged
    elif isinstance(existing, SimpleNamespace):
        merged = vars(existing).copy(); merged.update(cfg)
        impl._DEFAULT_CONFIG = SimpleNamespace(**merged)
    else:
        impl._DEFAULT_CONFIG = SimpleNamespace(**cfg)
