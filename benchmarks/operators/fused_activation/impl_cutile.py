"""cuTile fused element-wise mul-add + SiLU: out = silu(x * gate + bias).

1D grid, each CTA loads TILE elements. silu(z) implemented as
z / (1 + exp(-z)) since cuTile does not expose a fused sigmoid
primitive (cf. impl_triton.py which uses tl.sigmoid).
"""
from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

# Mirrors impl_triton.py via nw * occ ~= 64 (Triton sweeps nw in
# [2, 4, 8]; cuTile sweeps occ in [4, 8, 16, 32], adding occ=4 as the
# extra low-warps endpoint with no Triton counterpart).
_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [512, 1024, 2048]
    for occ in [4, 8, 16, 32]
]


@ct.kernel
def _fused_activation_kernel(x, gate, bias, out, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.astype(
        ct.load(x, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    gate_tile = ct.astype(
        ct.load(gate, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    bias_tile = ct.astype(
        ct.load(bias, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    z = x_tile * gate_tile + bias_tile
    out_tile = z / (1.0 + ct.exp(-z))  # SiLU = z * sigmoid(z) = z / (1 + exp(-z))
    ct.store(out, index=(bid,), tile=out_tile)


_tuner = CutileAutotuner(_fused_activation_kernel)


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    if x.shape != gate.shape or x.shape != bias.shape:
        raise ValueError("All input tensors must have the same shape.")

    x = x.contiguous()
    gate = gate.contiguous()
    bias = bias.contiguous()
    out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    n = x.numel()
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(n,),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: ((n + cfg.tile - 1) // cfg.tile, 1, 1),
            args_fn=lambda cfg: (x, gate, bias, out, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(tile=cfg.tile, occupancy=cfg.occupancy)
    else:
        cfg = _DEFAULT_CONFIG

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    grid = ((n + cfg.tile - 1) // cfg.tile, 1, 1)
    ct.launch(stream, grid, kernel, (x, gate, bias, out, cfg.tile))
    return out


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
