"""Shared helpers for the TileBench NCU scripts."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace


BACKENDS = ("triton", "cutile", "tilelang")
LEGACY_BACKENDS = ("triton", "cutile")


def repo_root() -> Path:
    """Resolve the TileBench repository root.

    Defaults to the parent of this file's directory, with TILEBENCH_ROOT as an
    escape hatch for remote boxes or copied scripts.
    """
    return Path(os.environ.get("TILEBENCH_ROOT", Path(__file__).resolve().parents[1])).resolve()


def ncu_bin() -> str:
    """Resolve the Nsight Compute CLI binary."""
    return os.environ.get("NCU_BIN") or shutil.which("ncu") or "/usr/local/cuda/bin/ncu"


def parse_backends(raw: str | None, *, default=BACKENDS) -> tuple[str, ...]:
    if not raw:
        return tuple(default)
    names = tuple(x.strip() for x in raw.split(",") if x.strip())
    bad = sorted(set(names) - set(BACKENDS))
    if bad:
        raise SystemExit(f"unknown NCU backend(s): {bad}; expected one of {BACKENDS}")
    return names


_PREFIXED_CONFIG_RE = re.compile(r"^_DEFAULT_([A-Z0-9]+)_CONFIG$")


def _inject_prefixed(impl, cfg: dict) -> dict:
    """Route prefixed winner keys into per-kernel config dicts."""
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


def apply_config_override(impl, cfg: dict | None, dtype_key=None) -> None:
    """Apply an autotune-winner config to an implementation module."""
    if not cfg:
        return
    cfg = _inject_prefixed(impl, dict(cfg))
    if not cfg:
        return

    existing = getattr(impl, "_DEFAULT_CONFIG", None)
    configs = getattr(impl, "_DEFAULT_CONFIGS", None)

    if existing is None and configs is not None:
        if dtype_key in configs:
            cur = configs[dtype_key]
            if isinstance(cur, dict):
                merged = dict(cur)
                merged.update(cfg)
                configs[dtype_key] = merged
            else:
                merged = vars(cur).copy()
                merged.update(cfg)
                configs[dtype_key] = SimpleNamespace(**merged)
        else:
            configs[dtype_key] = SimpleNamespace(**cfg)
    elif isinstance(existing, dict):
        merged = dict(existing)
        merged.update(cfg)
        impl._DEFAULT_CONFIG = merged
    elif isinstance(existing, SimpleNamespace):
        merged = vars(existing).copy()
        merged.update(cfg)
        impl._DEFAULT_CONFIG = SimpleNamespace(**merged)
    else:
        impl._DEFAULT_CONFIG = SimpleNamespace(**cfg)
