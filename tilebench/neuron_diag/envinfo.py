"""Environment snapshot for a diagnostics run.

Records versions and device configuration, never secrets: environment variables
are limited to Neuron/XLA/TileBench prefixes and any name that looks like a
credential is dropped entirely.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys

_PACKAGES = ("torch", "torch-xla", "torch-neuronx", "torch_neuronx", "nki", "neuronx-cc",
             "libneuronxla", "numpy", "triton")
_ENV_PREFIXES = ("NEURON_", "XLA_", "PJRT_", "TILEBENCH_", "UNSAFE_FP8FNCAST")
_SECRET = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|_KEY$|^AWS_", re.I)
# Values that identify the machine/account but are not needed for reproduction.
_ENV_SKIP = ("NEURON_RT_ROOT_COMM_ID",)


def _run(cmd: list[str], timeout: int = 30) -> str | None:
    if shutil.which(cmd[0]) is None and not os.path.isabs(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def package_versions() -> dict:
    out = {}
    for p in _PACKAGES:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def filtered_env() -> dict:
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith(_ENV_PREFIXES) and not _SECRET.search(k) and k not in _ENV_SKIP}


def neuron_system() -> dict:
    neuron_bin = "/opt/aws/neuron/bin"
    ls = _run([os.path.join(neuron_bin, "neuron-ls"), "--json-output"]) if os.path.isdir(neuron_bin) else None
    ls_text = _run([os.path.join(neuron_bin, "neuron-ls")]) if os.path.isdir(neuron_bin) else None
    dpkg = _run(["dpkg-query", "-W", "-f", "${Package} ${Version}\n"]) or ""
    driver = None
    try:
        with open("/sys/module/neuron/version") as f:
            driver = f.read().strip()
    except OSError:
        pass
    lnc = re.search(r"logical-neuroncore-config:\s*(\d+)", ls_text or "")
    itype = re.search(r"instance-type:\s*(\S+)", ls_text or "")
    try:
        devices = json.loads(ls) if ls else None
    except json.JSONDecodeError:
        devices = None
    if isinstance(devices, list):
        devices = [{k: v for k, v in d.items() if k not in ("neuron_processes",)} for d in devices]
    return {
        "instance_type": itype.group(1) if itype else None,
        "logical_neuroncore_config": int(lnc.group(1)) if lnc else None,
        "devices": devices,
        "driver_module_version": driver,
        "neuron_packages": sorted(l for l in dpkg.splitlines() if "neuron" in l.lower()),
    }


def container_info() -> dict:
    return {"in_container": os.path.exists("/.dockerenv"),
            "image_digest": os.environ.get("TILEBENCH_CONTAINER_IMAGE_DIGEST")}


def cpu_info() -> dict:
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    lscpu = _run(["lscpu"]) or ""
    keep = ("Architecture", "Model name", "CPU(s)", "NUMA node(s)", "NUMA node0 CPU(s)")
    return {"affinity": affinity,
            "lscpu": {l.split(":", 1)[0].strip(): l.split(":", 1)[1].strip()
                      for l in lscpu.splitlines() if ":" in l and l.split(":", 1)[0].strip() in keep}}


def neuron_processes() -> list[dict]:
    """Processes that hold a Neuron device (pid and command name only)."""
    out = _run(["/opt/aws/neuron/bin/neuron-ls", "--json-output"]) or ""
    try:
        devs = json.loads(out)
    except json.JSONDecodeError:
        return []
    procs = []
    for d in devs if isinstance(devs, list) else []:
        for p in d.get("neuron_processes") or []:
            procs.append({"pid": p.get("pid"), "command": os.path.basename(str(p.get("command", "")))[:60]})
    return procs


def snapshot(stack: str) -> dict:
    snap = {
        "stack": stack,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "neuron": neuron_system(),
        "neuron_processes": neuron_processes(),
        "container": container_info(),
        "cpu": cpu_info(),
        "env": filtered_env(),
    }
    return snap


def env_hash(snap: dict) -> str:
    """Hash of what changes a measurement's meaning: stack, packages, driver/runtime,
    device configuration and Neuron env flags (not transient process lists)."""
    key = {k: snap.get(k) for k in ("stack", "python", "packages", "env")}
    n = snap.get("neuron") or {}
    key["neuron"] = {k: n.get(k) for k in ("instance_type", "logical_neuroncore_config",
                                          "driver_module_version", "neuron_packages")}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
