from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile=256, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048]
    for occ in [4, 8, 16]
]
_last_autotune_config = None


@ct.kernel
def _max_pool2d_kernel(
    input_flat,
    output_flat,
    C,
    H,
    W,
    H_out,
    W_out,
    total_out,
    kernel_size: ConstInt,
    stride: ConstInt,
    padding: ConstInt,
    TILE: ConstInt,
):
    """
    Direct 2D stencil with max reduction, matching Triton's per-pixel-offset method:
      output[n, c, oh, ow] = max over (kh, kw) of
          input[n, c, oh*stride + kh - padding, ow*stride + kw - padding]

    Each block handles TILE consecutive output positions (flat layout).
    For each (kh, kw) kernel tap, compute the per-pixel flat input index and
    use ct.gather with padding_value=-inf — the cuTile equivalent of Triton's
    `tl.load(input_ptr + input_idx, mask=valid, other=-float("inf"))`.
    Invalid positions (spatial or output-range OOB) are clamped to -1 so
    ct.gather returns -inf and does not affect the max.
    """
    bid = ct.bid(0)
    offsets = bid * TILE + ct.arange(TILE, dtype=np.int32)
    mask = offsets < total_out

    # Decompose flat output offset into (n, c, oh, ow)
    ow = offsets % W_out
    oh = (offsets // W_out) % H_out
    c = (offsets // (H_out * W_out)) % C
    n = offsets // (C * H_out * W_out)

    acc = ct.full((TILE,), -float("inf"), dtype=np.float32)

    for kh in range(kernel_size):        # compile-time unrolled
        for kw in range(kernel_size):    # compile-time unrolled
            ih = oh * stride + kh - padding
            iw = ow * stride + kw - padding

            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            input_idx = ((n * C + c) * H + ih) * W + iw

            # Clamp invalid indices so ct.gather returns padding_value=-inf
            input_idx_safe = ct.where(valid, input_idx, -1)
            x = ct.gather(input_flat, input_idx_safe, padding_value=-float("inf"))
            x = ct.astype(x, np.float32)

            acc = ct.maximum(acc, x)

    # Cast back to output dtype before storing (avoids host-side conversion).
    acc = ct.astype(acc, output_flat.dtype)
    ct.store(output_flat, index=(bid,), tile=acc)


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(_max_pool2d_kernel)


def run(input, N, C, H, W, kernel_size, stride, padding,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    cuTile 2D max pooling — direct stencil matching Triton's method.
    input:  flat 1D tensor of size N * C * H * W
    output: flat 1D tensor of size N * C * H_out * W_out
    """
    global _last_autotune_config

    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    total_out = N * C * H_out * W_out

    if total_out <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(total_out, dtype=input.dtype, device=input.device)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(total_out, kernel_size, stride, padding),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (ct.cdiv(total_out, cfg.tile), 1, 1),
            args_fn=lambda cfg: (
                input, output,
                C, H, W, H_out, W_out, total_out,
                kernel_size, stride, padding,
                cfg.tile,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config = {
            "tile":      cfg.tile,
            "occupancy": cfg.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG

    grid = (ct.cdiv(total_out, cfg.tile), 1, 1)
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream, grid, kernel,
        (input, output,
         C, H, W, H_out, W_out, total_out,
         kernel_size, stride, padding,
         cfg.tile),
    )

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
