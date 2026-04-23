import math
from types import SimpleNamespace

import cuda.tile as ct
import torch
import torch.nn.functional as F

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:  # pragma: no cover
    ct_experimental = None

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(tile_m=64, tile_k=32, tile_n=64, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile_m=tm, tile_k=tk, tile_n=tn, occupancy=occ)
    for tm in [32, 64, 128]
    for tk in [16, 32, 64]
    for tn in [64, 128]
    for occ in [4, 8, 16, 32]
    if (
        tm * tn * occ <= 64 * 128 * 16
        and not (max(tm, tn) >= 128 and tk == 16)
    )
]

# All candidate TILE_* values must divide the corresponding MAX_TILE_*
# so the padded buffers stay in-bounds for any config.
_MAX_TILE_M = 128
_MAX_TILE_K = 64
_MAX_TILE_N = 128


@ct.kernel
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    K_TILES: ConstInt,
    TILE_M: ConstInt,
    TILE_K: ConstInt,
    TILE_N: ConstInt,
):
    """GEMM: C[M, N] = A[M, K] @ B[K, N]. Each block produces one (TILE_M, TILE_N) C tile."""
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    for bid_k in range(K_TILES):
        a_tile = ct.load(a_ptr, index=(bid_m, bid_k), shape=(TILE_M, TILE_K))
        b_tile = ct.load(b_ptr, index=(bid_k, bid_n), shape=(TILE_K, TILE_N))
        acc = acc + ct.matmul(a_tile, b_tile)
    ct.store(c_ptr, index=(bid_m, bid_n), tile=acc)


def run(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 1,
    groups: int = 1,
    block_size: int = 1024,
    autotune: bool = False,
    **kwargs,
):
    """
    cuTile Conv2d forward via im2col + cuTile GEMM.

    input:  (batch, in_channels, H, W)
    weight: (out_channels, in_channels // groups, kH, kW)
    output: (batch, out_channels, out_H, out_W)
    """
    global _last_autotune_config

    batch, in_channels, in_H, in_W = input.shape
    out_channels, in_channels_per_group, kH, kW = weight.shape
    assert in_channels_per_group * groups == in_channels

    out_H = (in_H + 2 * padding - kH) // stride + 1
    out_W = (in_W + 2 * padding - kW) // stride + 1
    out_channels_per_group = out_channels // groups

    stream = torch.cuda.current_stream()

    output_groups = []
    for g in range(groups):
        x_g = input[:, g * in_channels_per_group : (g + 1) * in_channels_per_group, :, :]
        w_g = weight[g * out_channels_per_group : (g + 1) * out_channels_per_group, :, :, :]

        col = F.unfold(
            x_g.float(),
            kernel_size=(kH, kW),
            padding=padding,
            stride=stride,
        )

        K_feat = in_channels_per_group * kH * kW
        L = out_H * out_W
        M = batch * L
        col_2d = col.permute(0, 2, 1).reshape(M, K_feat).contiguous()
        w_2d = w_g.reshape(out_channels_per_group, K_feat).t().contiguous()

        # Pad to per-axis MAX_TILE_* so any candidate config in _SEARCH_SPACE works.
        M_pad  = (M      + _MAX_TILE_M - 1) // _MAX_TILE_M * _MAX_TILE_M
        K_pad  = (K_feat + _MAX_TILE_K - 1) // _MAX_TILE_K * _MAX_TILE_K
        OC_pad = (out_channels_per_group + _MAX_TILE_N - 1) // _MAX_TILE_N * _MAX_TILE_N

        a_pad = torch.zeros((M_pad, K_pad),  device=input.device, dtype=torch.float32)
        b_pad = torch.zeros((K_pad, OC_pad), device=input.device, dtype=torch.float32)
        a_pad[:M,      :K_feat] = col_2d
        b_pad[:K_feat, :out_channels_per_group] = w_2d
        c_pad = torch.empty((M_pad, OC_pad), device=input.device, dtype=torch.float32)

        if autotune and ct_experimental is not None:
            result = ct_experimental.autotune_launch(
                stream,
                grid_fn=lambda cfg: (M_pad // cfg.tile_m, OC_pad // cfg.tile_n, 1),
                kernel=_gemm_kernel,
                args_fn=lambda cfg: (
                    a_pad, b_pad, c_pad,
                    K_pad // cfg.tile_k,
                    cfg.tile_m, cfg.tile_k, cfg.tile_n,
                ),
                hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
                search_space=_SEARCH_SPACE,
            )
            _last_autotune_config = {
                "tile_m": result.tuned_config.tile_m,
                "tile_k": result.tuned_config.tile_k,
                "tile_n": result.tuned_config.tile_n,
                "occupancy": result.tuned_config.occupancy,
            }
        else:
            cfg = _DEFAULT_CONFIG
            grid = (M_pad // cfg.tile_m, OC_pad // cfg.tile_n, 1)
            ct.launch(
                stream, grid, _gemm_kernel,
                (a_pad, b_pad, c_pad,
                 K_pad // cfg.tile_k,
                 cfg.tile_m, cfg.tile_k, cfg.tile_n),
            )

        out_g = c_pad[:M, :out_channels_per_group]
        out_g = out_g.reshape(batch, L, out_channels_per_group).permute(0, 2, 1)
        out_g = out_g.reshape(batch, out_channels_per_group, out_H, out_W)
        output_groups.append(out_g)

    output = torch.cat(output_groups, dim=1)
    return output.to(input.dtype)


def get_last_config() -> dict | None:
    return _last_autotune_config
