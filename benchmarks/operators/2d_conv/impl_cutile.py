import math

import cuda.tile as ct
import torch
import torch.nn.functional as F

ConstInt = ct.Constant[int]


def _tile_dim(block_size: int) -> int:
    root = int(math.sqrt(block_size))
    tile = 16
    while tile * 2 <= root:
        tile *= 2
    return max(16, tile)


@ct.kernel
def _gemm_kernel(a_ptr, b_ptr, c_ptr, K_TILES: ConstInt, TILE: ConstInt):
    """Simple GEMM kernel reused for the im2col matmul."""
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    acc = ct.zeros((TILE, TILE), dtype=ct.float32)
    for bid_k in range(K_TILES):
        a_tile = ct.load(a_ptr, index=(bid_m, bid_k), shape=(TILE, TILE))
        b_tile = ct.load(b_ptr, index=(bid_k, bid_n), shape=(TILE, TILE))
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
    batch, in_channels, in_H, in_W = input.shape
    out_channels, in_channels_per_group, kH, kW = weight.shape
    assert in_channels_per_group * groups == in_channels

    out_H = (in_H + 2 * padding - kH) // stride + 1
    out_W = (in_W + 2 * padding - kW) // stride + 1
    out_channels_per_group = out_channels // groups

    tile = _tile_dim(block_size)
    stream = torch.cuda.current_stream()

    # Process each group independently (groups=1 is the common case)
    output_groups = []
    for g in range(groups):
        # Slice along the channel dimension for this group
        x_g = input[:, g * in_channels_per_group : (g + 1) * in_channels_per_group, :, :]
        w_g = weight[g * out_channels_per_group : (g + 1) * out_channels_per_group, :, :, :]

        # im2col: unfold x_g → (batch, in_channels_per_group * kH * kW, out_H * out_W)
        # F.unfold expects (batch, C, H, W) and returns (batch, C*kH*kW, out_H*out_W)
        col = F.unfold(
            x_g.float(),
            kernel_size=(kH, kW),
            padding=padding,
            stride=stride,
        )  # (batch, K, L) where K = in_channels_per_group*kH*kW, L = out_H*out_W

        K_feat = in_channels_per_group * kH * kW
        L = out_H * out_W

        # Reshape to (M, K) where M = batch * L
        M = batch * L
        col_2d = col.permute(0, 2, 1).reshape(M, K_feat).contiguous()  # (M, K)

        # Flatten weight to (out_channels_per_group, K)
        w_2d = w_g.reshape(out_channels_per_group, K_feat).t().contiguous()  # (K, OC)

        # Pad to tile multiples
        M_pad  = (M      + tile - 1) // tile * tile
        K_pad  = (K_feat + tile - 1) // tile * tile
        OC_pad = (out_channels_per_group + tile - 1) // tile * tile

        a_pad = torch.zeros((M_pad, K_pad),  device=input.device, dtype=torch.float32)
        b_pad = torch.zeros((K_pad, OC_pad), device=input.device, dtype=torch.float32)
        a_pad[:M,      :K_feat] = col_2d
        b_pad[:K_feat, :out_channels_per_group] = w_2d

        c_pad = torch.empty((M_pad, OC_pad), device=input.device, dtype=torch.float32)
        grid  = (M_pad // tile, OC_pad // tile, 1)
        ct.launch(
            stream, grid, _gemm_kernel,
            (a_pad, b_pad, c_pad, K_pad // tile, tile),
        )

        # Trim and reshape → (batch, out_channels_per_group, out_H, out_W)
        out_g = c_pad[:M, :out_channels_per_group]            # (M, OC)
        out_g = out_g.reshape(batch, L, out_channels_per_group).permute(0, 2, 1)  # (batch, OC, L)
        out_g = out_g.reshape(batch, out_channels_per_group, out_H, out_W)
        output_groups.append(out_g)

    output = torch.cat(output_groups, dim=1)  # (batch, out_channels, out_H, out_W)
    return output.to(input.dtype)


def get_last_config() -> dict | None:
    return None
