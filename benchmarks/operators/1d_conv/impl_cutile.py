from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:  # pragma: no cover
    ct_experimental = None

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048]
    for occ in [4, 8, 16]
]
_last_autotune_config = None


@ct.kernel
def _conv1d_stencil_kernel(
    input_flat,
    kernel_flat,
    output_flat,
    kernel_size: ConstInt,
    TILE: ConstInt,
):
    """
    Direct 1D stencil matching Triton's method exactly:
      acc[i] = sum_{j=0..kernel_size-1} input[i+j] * kernel[j]

    Each block handles TILE consecutive output elements. For each j, use
    ct.gather for runtime-indexed per-element loads — the cuTile equivalent
    of Triton's `tl.load(input_ptr + offsets + j)` per-pixel pointer
    arithmetic. ct.gather's padding_value=0 handles tail-tile OOB reads;
    ct.store silently drops OOB writes for the final partial output tile.
    """
    bid = ct.bid(0)
    offsets = bid * TILE + ct.arange(TILE, dtype=np.int32)

    acc = ct.zeros((TILE,), dtype=np.float32)

    for j in range(kernel_size):  # compile-time unrolled
        input_idx = offsets + j
        x = ct.gather(input_flat, input_idx, padding_value=0.0)
        x = ct.astype(x, np.float32)

        w_scalar = ct.load(kernel_flat, index=(j,), shape=())
        w_scalar = ct.astype(w_scalar, np.float32)

        acc = acc + x * w_scalar

    ct.store(output_flat, index=(bid,), tile=acc)


def run(input, kernel, input_size, kernel_size,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    cuTile 1D valid convolution — direct stencil matching Triton's method.
    input:  flat 1D tensor of size input_size
    kernel: flat 1D tensor of size kernel_size
    output: flat 1D tensor of size input_size - kernel_size + 1
    """
    global _last_autotune_config

    output_size = input_size - kernel_size + 1
    if output_size <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(output_size, dtype=torch.float32, device=input.device)
    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (ct.cdiv(output_size, cfg.tile), 1, 1),
            kernel=_conv1d_stencil_kernel,
            args_fn=lambda cfg: (
                input, kernel, output,
                kernel_size,
                cfg.tile,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {
            "tile":      result.tuned_config.tile,
            "occupancy": result.tuned_config.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG
        grid = (ct.cdiv(output_size, cfg.tile), 1, 1)
        ct.launch(
            stream, grid, _conv1d_stencil_kernel,
            (input, kernel, output, kernel_size, cfg.tile),
        )

    return output.to(input.dtype)


def get_last_config() -> dict | None:
    return _last_autotune_config
