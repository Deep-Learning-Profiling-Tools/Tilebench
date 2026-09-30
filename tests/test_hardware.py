"""tilebench.hardware: architecture detection from the actual device, and the
cache-eviction size the timer derives from it.

Every device here is mocked, so the tests run on any host (no GPU needed) and
never depend on real GH200 / MI300X hardware.
"""
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tilebench import hardware
from tilebench.core import timer

MB = 1024 * 1024
B200_L2 = 132644864          # torch.cuda.get_device_properties(0).L2_cache_size on B200


@pytest.fixture
def device(monkeypatch):
    """Install a fake CUDA/HIP device; returns a setter."""
    def install(*, available=True, name="GPU", capability=None, hip=None, gcn=None, l2=0,
                raises=False):
        hardware.device_info.cache_clear()
        hardware.detect_arch.cache_clear()
        props = SimpleNamespace(name=name, L2_cache_size=l2)
        if gcn is not None:
            props.gcnArchName = gcn

        def get_props(_index):
            if raises:
                raise RuntimeError("no device")
            return props
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "get_device_properties", get_props)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _i=0: capability)
        monkeypatch.setattr(torch.version, "hip", hip)
    yield install
    hardware.device_info.cache_clear()
    hardware.detect_arch.cache_clear()


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def test_no_gpu_is_none_and_supports_nothing(device):
    device(available=False)
    assert hardware.device_info() is None
    assert hardware.detect_arch() is None
    assert not hardware.supports_tma() and not hardware.supports_tmem()


def test_a_failing_device_query_is_none_not_an_exception(device):
    device(capability=(10, 0), raises=True)
    assert hardware.device_info() is None and hardware.detect_arch() is None


def test_blackwell(device):
    device(name="NVIDIA B200", capability=(10, 0), l2=B200_L2)
    assert hardware.detect_arch() == "blackwell"
    assert hardware.supports_tmem() and hardware.supports_tma()
    assert hardware.device_info() == hardware.DeviceInfo("nvidia", "NVIDIA B200", (10, 0), None)


def test_hopper(device):
    device(name="NVIDIA GH200 480GB", capability=(9, 0))
    assert hardware.detect_arch() == "hopper"
    assert hardware.supports_tma() and not hardware.supports_tmem()


def test_amd_gfx942_is_cdna3(device):
    device(name="AMD Instinct MI300X", hip="6.2.41133", gcn="gfx942:sramecc+:xnack-")
    assert hardware.detect_arch() == "cdna3"
    assert not hardware.supports_tma() and not hardware.supports_tmem()
    assert hardware.device_info().gcn_arch_name == "gfx942:sramecc+:xnack-"
    assert hardware.device_info().capability is None


@pytest.mark.parametrize("kwargs", [dict(capability=(8, 0)), dict(capability=(12, 0)),
                                    dict(hip="6.2", gcn="gfx90a:sramecc+:xnack-"),
                                    dict(hip="6.2", gcn=None)])
def test_devices_outside_the_scope_select_no_architecture_path(device, kwargs):
    device(**kwargs)
    assert hardware.device_info() is not None           # still reported
    assert hardware.detect_arch() is None
    assert not hardware.supports_tma() and not hardware.supports_tmem()


def test_the_result_label_is_not_an_input():
    """--gpu names the result namespace only: nothing in the detection API takes it."""
    import inspect
    for fn in (hardware.detect_arch, hardware.device_info, hardware.supports_tma,
               hardware.supports_tmem, hardware.last_level_cache_bytes):
        assert not inspect.signature(fn).parameters


def test_import_does_not_initialise_the_gpu():
    code = ("import torch, tilebench.hardware, tilebench.core.timer; "
            "print(torch.cuda.is_initialized())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"


# --------------------------------------------------------------------------
# cache eviction: B200 behaviour is unchanged
# --------------------------------------------------------------------------

def flush_mb_before_refactor(l2_bytes=None):
    """timer._flush_l2_buffer_mb as it was before the refactor (origin/main
    058c0f2f), with the device query replaced by its outcome."""
    if l2_bytes is None:                  # the query raised
        l2_bytes = 128 * 1024 * 1024
    return max(64, 2 * l2_bytes // (1024 * 1024))


def test_b200_flush_size_is_unchanged(device):
    device(name="NVIDIA B200", capability=(10, 0), l2=B200_L2)
    assert hardware.last_level_cache_bytes() == B200_L2
    assert timer._flush_l2_buffer_mb() == flush_mb_before_refactor(B200_L2) == 253


@pytest.mark.parametrize("capability,l2", [((9, 0), 50 * MB), ((8, 0), 40 * MB), ((8, 0), 0),
                                           ((12, 0), 96 * MB)])
def test_nvidia_flush_size_is_unchanged(device, capability, l2):
    device(capability=capability, l2=l2)
    assert timer._flush_l2_buffer_mb() == flush_mb_before_refactor(l2)


def test_failed_query_keeps_the_old_fallback(device):
    device(capability=(10, 0), raises=True)
    assert hardware.last_level_cache_bytes() is None
    assert timer._flush_l2_buffer_mb() == flush_mb_before_refactor(None) == 256


def test_a_measured_llc_replaces_the_runtime_l2(device, monkeypatch):
    """How an architecture whose L2 is not the last level (CDNA3) plugs in its
    measured cache size; no value is registered until it has been measured."""
    assert hardware._LLC_BYTES == {}
    device(hip="6.2", gcn="gfx942:sramecc+:xnack-", l2=4 * MB)
    assert timer._flush_l2_buffer_mb() == flush_mb_before_refactor(4 * MB)   # runtime L2 today
    monkeypatch.setitem(hardware._LLC_BYTES, "cdna3", 300 * MB)   # synthetic, not a spec value
    assert hardware.last_level_cache_bytes() == 300 * MB
    assert timer._flush_l2_buffer_mb() == 600
