"""Architecture of the GPU this process actually runs on.

Operators that need an architecture-specific code path (TMEM, host-side TMA)
branch on these helpers, never on the ``--gpu`` result label: ``--gpu`` only
names the result namespace ``results/<gpu>/``, so a label that does not match
the machine would otherwise silently select the wrong kernels.

Nothing here touches the GPU at import time. The device is probed on first use
and cached; on a host without a CUDA/HIP GPU (e.g. an AWS Trainium host running
NKI) every probe returns ``None`` / ``False`` instead of raising.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

#: Architectures in TileBench++ scope. A device outside these maps is reported
#: by device_info() but detect_arch() returns None for it, so no
#: architecture-specific path is ever selected for hardware nobody validated.
_NVIDIA_ARCH = {(10, 0): "blackwell", (9, 0): "hopper"}
_AMD_ARCH = {"gfx942": "cdna3"}

#: Device-level last-level cache size in bytes, for architectures where the
#: runtime's L2_cache_size is not the cache the timer must evict (e.g. the
#: Infinity Cache of CDNA3). Filled only with a value validated by an eviction
#: sweep on the hardware itself; an architecture without an entry uses the
#: runtime-reported L2 size, unless it is listed in _LLC_CALIBRATION_REQUIRED.
#:
#: cdna3: MI300X (gfx942:sramecc+:xnack-, ROCm 7.1). The runtime reports a 4 MiB
#: L2 and a 256 MiB L3 (Infinity Cache) shared by all 304 CUs (rocminfo,
#: amd-smi, KFD topology). Eviction sweep with timer._flush_l2_cache before each
#: Proton-timed call of two HBM-bound probes (Triton add over 48 MiB, torch
#: elementwise over 64 MiB), eager and CUDA-graph, randomized order, 10 rounds:
#: <= 192 MiB leaves the probes warm, 256 MiB (1x L3) evicts only sometimes
#: (round medians spread up to 38%), and from 384 MiB they are cold, with
#: 512 and 768 MiB agreeing within 1.3%. The timer's 2x rule gives 512 MiB.
_LLC_BYTES: dict[str, int] = {"cdna3": 256 * 1024 * 1024}

#: Architectures whose runtime L2 is known not to be the last-level cache: a
#: cold-cache measurement refuses to run on them until _LLC_BYTES holds their
#: validated size, instead of silently evicting only the L2.
_LLC_CALIBRATION_REQUIRED = frozenset({"cdna3"})


class UncalibratedCacheError(RuntimeError):
    """The eviction size of this architecture's last-level cache is unknown."""


class DeviceInfo(NamedTuple):
    vendor: str                           # "nvidia" or "amd"
    name: str
    capability: tuple[int, int] | None   # NVIDIA compute capability
    gcn_arch_name: str | None            # AMD gcnArchName, e.g. "gfx942:sramecc+:xnack-"


@functools.cache
def device_info() -> DeviceInfo | None:
    """The current CUDA/HIP device, or None when there is none."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        if torch.version.hip:
            return DeviceInfo("amd", props.name, None, getattr(props, "gcnArchName", None) or None)
        return DeviceInfo("nvidia", props.name, tuple(torch.cuda.get_device_capability(index)), None)
    except Exception:
        return None


@functools.cache
def detect_arch() -> str | None:
    """"blackwell" (sm_100), "hopper" (sm_90), "cdna3" (gfx942), or None
    (no CUDA/HIP GPU, or a device outside TileBench++ scope)."""
    dev = device_info()
    if dev is None:
        return None
    if dev.vendor == "nvidia":
        return _NVIDIA_ARCH.get(dev.capability)
    return _AMD_ARCH.get((dev.gcn_arch_name or "").split(":")[0])


def supports_tmem() -> bool:
    """Tensor memory (T.alloc_tmem / tcgen05): Blackwell only."""
    return detect_arch() == "blackwell"


def supports_tma() -> bool:
    """Host-side TMA tensor descriptors: NVIDIA Hopper and Blackwell."""
    return detect_arch() in ("hopper", "blackwell")


def last_level_cache_bytes() -> int | None:
    """Size of the device-level last-level cache that must be evicted for a
    cold-entry measurement, or None when it cannot be determined.

    Raises UncalibratedCacheError on an architecture listed in
    _LLC_CALIBRATION_REQUIRED that has no validated _LLC_BYTES entry."""
    arch = detect_arch()
    if arch in _LLC_BYTES:
        return _LLC_BYTES[arch]
    if arch in _LLC_CALIBRATION_REQUIRED:
        raise UncalibratedCacheError(
            f"{arch.upper()} LLC eviction size has not been calibrated yet. Its runtime L2 is "
            f"not the last-level cache, so evicting it would not give a cold-cache measurement. "
            f"Validate the size with an eviction sweep on the hardware and register it in "
            f"tilebench.hardware._LLC_BYTES[{arch!r}] before any flushed performance run.")
    try:
        import torch
        return torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    except Exception:
        return None
