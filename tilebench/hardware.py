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
#: Infinity Cache of CDNA3). Empty until measured on the hardware itself; an
#: architecture without an entry uses the runtime-reported L2 size.
_LLC_BYTES: dict[str, int] = {}


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
    cold-entry measurement, or None when it cannot be determined."""
    arch = detect_arch()
    if arch in _LLC_BYTES:
        return _LLC_BYTES[arch]
    try:
        import torch
        return torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    except Exception:
        return None
