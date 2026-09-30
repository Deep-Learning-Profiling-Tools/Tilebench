"""Provenance of a benchmark run: which source, software stack and device
produced it.

collect() is called once, before any measurement, and returns a JSON-ready
dict. It never raises: a field that cannot be determined (no git, no GPU,
backend not installed) is None, and a source-control failure is described in
source["error"].

Where it is stored (the result JSON formats are unchanged, so every existing
reader keeps working):
  - run_bench.py writes a sidecar with the same file name as the run's timing
    and autotune logs, in its own directory:
        results/<gpu>/logs/provenance/<op>_<mode>_<backends>.json
    (tilebench.paths.provenance_log_path), or next to an explicit --output as
    <output stem>.provenance.json; its "run" block names the logs and the
    summary CSV it describes.
  - run_bench_all.py adds it under the "provenance" key of the run's
    results/<gpu>/runs/<timestamp>/summary.json.
"""

from __future__ import annotations

import importlib.metadata
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tilebench import hardware
from tilebench.paths import PACKAGE_ROOT

SCHEMA = "tilebench-provenance/1"

#: Tracked files under these paths are benchmark output, not source: a run that
#: rewrites results/<gpu>/csv/ does not make the next run's source "dirty".
_DATA_PATHS = ("results",)
_MAX_LISTED = 50


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True, timeout=60).stdout


def source_state(repo: Path = PACKAGE_ROOT) -> dict:
    """Commit and working-tree state of the checkout holding ``repo``.

    dirty: a tracked file outside results/ differs from HEAD (staged or not).
    Untracked files do not make the tree dirty; they are listed separately."""
    try:
        top = Path(_git(repo, "rev-parse", "--show-toplevel").strip())
        sha = _git(top, "rev-parse", "HEAD").strip()
        excludes = [f":(exclude){p}" for p in _DATA_PATHS]
        changed = [line[3:] for line in _git(
            top, "status", "--porcelain", "--untracked-files=no", "--", ".", *excludes
        ).splitlines()]
        untracked = [line[3:] for line in _git(
            top, "status", "--porcelain", "--untracked-files=all", "--", ".", *excludes
        ).splitlines() if line.startswith("??")]
    except Exception as e:  # no git, not a checkout, ...
        return {"git_sha": None, "dirty": None, "error": f"{type(e).__name__}: {e}".strip()}
    return {"git_sha": sha, "dirty": bool(changed),
            "dirty_files": changed[:_MAX_LISTED], "untracked_files": untracked[:_MAX_LISTED]}


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_version(module: str) -> str | None:
    """Version of whichever distribution provides ``module`` (Triton ships as
    ``triton`` on CUDA and under another distribution name on ROCm)."""
    try:
        dists = importlib.metadata.packages_distributions().get(module, [])
    except Exception:
        return None
    for dist in dists:
        version = _distribution_version(dist)
        if version:
            return version
    return None


def software_state() -> dict:
    state = {"python": platform.python_version(), "torch": None, "torch_cuda": None,
             "torch_hip": None}
    try:
        import torch
        state.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                     torch_hip=torch.version.hip)
    except Exception:
        pass
    state["triton"] = _module_version("triton")
    state["tilelang"] = _module_version("tilelang")
    # cuda.tile is a namespace package shared with other cuda-* distributions,
    # so it is looked up by distribution name.
    state["cuda_tile"] = _distribution_version("cuda-tile")
    return state


def device_state(requested_label: str) -> dict:
    dev = hardware.device_info()
    return {
        "requested_label": requested_label,
        "name": dev.name if dev else None,
        "vendor": dev.vendor if dev else None,
        "arch": hardware.detect_arch(),
        "compute_capability": list(dev.capability) if dev and dev.capability else None,
        "gcn_arch_name": dev.gcn_arch_name if dev else None,
    }


def collect(requested_label: str) -> dict:
    """Provenance of a run whose results go to results/<requested_label>/."""
    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source_state(),
        "software": software_state(),
        "device": device_state(requested_label),
        "host": {"hostname": socket.gethostname(), "argv": list(sys.argv)},
    }
