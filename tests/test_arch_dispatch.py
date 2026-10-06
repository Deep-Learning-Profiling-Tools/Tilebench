"""Architecture-specific operator paths: which kernel body / default config each
architecture gets, and that the tuning candidates never depend on it.

The hardware probes and the host architecture are mocked and the operator
modules re-imported, so every case runs on any host (no GPU needed) except the
one end-to-end int8 check, which needs a CUDA device.
"""
import importlib
import platform

import pytest
import torch

from tilebench import hardware

OPS = "tilebench.benchmarks.operators"

# operator -> (kernel attribute(s), candidate-list function or None for plain @tilelang.jit)
TMEM_KERNELS = {
    "batched_matmul": (["bmm_kernel"], "bmm_configs"),
    "matmul_fp32_fp16_fp8": (["matmul_kernel"], "matmul_configs"),
    "matmul_int8": (["matmul_kernel"], "matmul_configs"),
    "streamk_matmul": (["first_wave_kernel", "full_tiles_kernel"], None),
    "1d_conv": (["conv1d_kernel"], "conv1d_configs"),
    "2d_conv": (["conv2d_kernel"], "conv2d_configs"),
    "3d_conv": (["conv3d_kernel"], "conv3d_configs"),
    "flash_attention": (["flash_attention_kernel"], "flash_attention_configs"),
    "block_sparse_attention": (["block_sparse_attention_kernel"], "block_sparse_attention_configs"),
}


def _jit(kernel):
    """The @tilelang.jit object under an optional @tilelang.autotune."""
    return getattr(kernel, "jit_impl", kernel)


@pytest.fixture
def load(monkeypatch):
    """Import an operator module under a patched environment; re-import it
    unpatched afterwards so later tests see the real module."""
    loaded = []

    def _load(name, **patches):
        for target, value in patches.items():
            monkeypatch.setattr(target, value)
        mod = importlib.reload(importlib.import_module(f"{OPS}.{name}"))
        loaded.append(mod)
        return mod
    yield _load
    monkeypatch.undo()
    for mod in loaded:
        importlib.reload(mod)


# --------------------------------------------------------------------------
# TileLang: TMEM kernels on Blackwell, register-fragment kernels elsewhere
# --------------------------------------------------------------------------

@pytest.mark.parametrize("op", sorted(TMEM_KERNELS))
@pytest.mark.parametrize("tmem", [True, False], ids=["blackwell", "hopper"])
def test_tilelang_kernel_body_follows_tmem_support(load, op, tmem):
    mod = load(f"{op}.impl_tilelang", **{"tilebench.hardware.supports_tmem": lambda: tmem})
    names, configs_fn = TMEM_KERNELS[op]
    for name in names:
        kernel = getattr(mod, name)
        jit = _jit(kernel)
        assert jit.func.__name__ == name                     # public kernel name unchanged
        assert ("T.alloc_tmem" in jit.func_source) is tmem
        if not tmem:
            assert "mbar=" not in jit.func_source
        if configs_fn is not None:                           # identical candidate list
            assert kernel.configs == getattr(mod, configs_fn)()


@pytest.mark.parametrize("op", sorted(TMEM_KERNELS))
def test_tilelang_candidates_do_not_depend_on_architecture(load, op):
    names, configs_fn = TMEM_KERNELS[op]
    if configs_fn is None:
        pytest.skip("tuned by the operator's own pipeline, not @tilelang.autotune")
    bw = load(f"{op}.impl_tilelang", **{"tilebench.hardware.supports_tmem": lambda: True})
    bw_configs = getattr(bw, names[0]).configs
    hp = load(f"{op}.impl_tilelang", **{"tilebench.hardware.supports_tmem": lambda: False})
    assert getattr(hp, names[0]).configs == bw_configs


# --------------------------------------------------------------------------
# Triton streamk_matmul: default-config legality on Hopper fp32
# --------------------------------------------------------------------------

_STREAMK_DEFAULT = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8,
                    "num_warps": 8, "num_stages": 3}
_STREAMK_SEARCH_SPACE = [
    {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8, "num_warps": nw, "num_stages": 3}
    for bm in (64, 128) for bn in (128, 256) for bk in (32, 64) for nw in (4, 8)
]


@pytest.fixture
def streamk(load):
    def _get(arch):
        mod = load("streamk_matmul.impl_triton")
        mod.detect_arch = lambda: arch                       # restored by the reload in `load`
        return mod
    return _get


def test_streamk_search_space_and_default_are_unchanged(streamk):
    mod = streamk("hopper")
    assert mod._SEARCH_SPACE == _STREAMK_SEARCH_SPACE
    assert mod._DEFAULT_CONFIG == _STREAMK_DEFAULT


@pytest.mark.parametrize("arch", ["blackwell", None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_streamk_default_is_the_original_off_hopper_and_cdna3(streamk, arch, dtype):
    mod = streamk(arch)
    assert mod._default_config(dtype) == _STREAMK_DEFAULT


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, None])
def test_streamk_cdna3_fallback_covers_every_dtype(streamk, dtype):
    mod = streamk("cdna3")
    assert mod._default_config(dtype) is mod._CDNA3_DEFAULT_CONFIG


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_streamk_hopper_half_precision_keeps_the_original_default(streamk, dtype):
    assert streamk("hopper")._default_config(dtype) == _STREAMK_DEFAULT


def test_streamk_hopper_fp32_only_lowers_block_k(streamk):
    cfg = streamk("hopper")._default_config(torch.float32)
    assert cfg == {**_STREAMK_DEFAULT, "BLOCK_K": 32}
    assert cfg in _STREAMK_SEARCH_SPACE


@pytest.mark.parametrize("arch", ["hopper", "cdna3", "blackwell"])
def test_streamk_replayed_default_override_is_used_as_given(streamk, arch):
    mod = streamk(arch)
    replay = {**_STREAMK_DEFAULT, "BLOCK_M": 64}              # what the NCU harness writes
    mod._DEFAULT_CONFIG = replay
    assert mod._default_config(torch.float32) is replay


# --------------------------------------------------------------------------
# TileLang dequantize_rowwise: explicit int8 sign extension on aarch64 hosts
# --------------------------------------------------------------------------

@pytest.mark.parametrize("machine, explicit", [("x86_64", False), ("aarch64", True), ("arm64", True)])
def test_dequantize_int8_path_follows_host_architecture(load, machine, explicit):
    mod = load("dequantize_rowwise.impl_tilelang", **{"platform.machine": lambda: machine})
    assert mod._EXPLICIT_INT8_SIGN_EXTENSION is explicit
    jit = _jit(mod.dequantize_rowwise_kernel)
    assert jit.func.__name__ == "dequantize_rowwise_kernel"
    assert ("T.reinterpret" in jit.func_source) is explicit
    assert mod.dequantize_rowwise_kernel.configs == mod.dequantize_rowwise_configs()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_dequantize_every_int8_value_keeps_its_sign():
    mod = importlib.import_module(f"{OPS}.dequantize_rowwise.impl_tilelang")
    ref_mod = importlib.import_module(f"{OPS}.dequantize_rowwise.impl_torch")
    idx = torch.arange(512 * 512, device="cuda").view(512, 512)
    x = ((idx % 256) - 128).to(torch.int8)                  # -128 .. 127, each many times
    state = torch.rand(512, device="cuda") * 10.0 + 0.5
    out = mod.run(x, state).float()
    torch.testing.assert_close(out, ref_mod.run(x, state).float(), atol=1e-3, rtol=1e-3)
    for v in (-128, -1, 0, 1, 127):
        got = out[x == v]
        assert bool((torch.sign(got) == (1 if v > 0 else -1 if v < 0 else 0)).all()), v
