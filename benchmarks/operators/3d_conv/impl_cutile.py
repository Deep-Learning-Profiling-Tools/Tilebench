from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:  # pragma: no cover
    ct_experimental = None

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=4)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048]
    for occ in [2, 4, 8, 16, 32]
]
_last_autotune_config = None


@ct.kernel
def _conv3d_stencil_kernel(
    input_flat,
    kernel_flat,
    output_flat,
    input_rows,
    input_cols,
    output_rows,
    output_cols,
    total_out,
    kernel_depth: ConstInt,
    kernel_rows: ConstInt,
    kernel_cols: ConstInt,
    TILE: ConstInt,
):
    """
    Direct 3D stencil matching Triton's per-pixel-offset method:
      output[od, or, oc] = sum_{kd,kr,kc} input[od+kd, or+kr, oc+kc] * kernel[kd, kr, kc]

    Each block handles TILE consecutive output positions (flat 1D layout).
    For each (kd, kr, kc) kernel tap, compute the per-pixel flat input
    index and use ct.gather — the cuTile equivalent of Triton's
    `tl.load(input_ptr + input_idx, mask=mask, other=0.0)`.
    ct.gather's padding_value handles the last-tile tail; ct.store
    silently drops OOB writes.
    """
    bid = ct.bid(0)
    offsets = bid * TILE + ct.arange(TILE, dtype=np.int32)

    # Decompose flat output offset into (od, or, oc)
    output_plane = output_rows * output_cols
    od = offsets // output_plane
    remain = offsets % output_plane
    oh = remain // output_cols
    ow = remain % output_cols

    input_plane = input_rows * input_cols
    kernel_plane = kernel_rows * kernel_cols

    acc = ct.zeros((TILE,), dtype=np.float32)

    for kd in range(kernel_depth):          # compile-time unrolled
        for kr in range(kernel_rows):       # compile-time unrolled
            for kc in range(kernel_cols):   # compile-time unrolled
                input_idx = (od + kd) * input_plane + (oh + kr) * input_cols + (ow + kc)
                x = ct.gather(input_flat, input_idx, padding_value=0.0)
                x = ct.astype(x, np.float32)

                kernel_idx = kd * kernel_plane + kr * kernel_cols + kc
                w_scalar = ct.load(kernel_flat, index=(kernel_idx,), shape=())
                w_scalar = ct.astype(w_scalar, np.float32)

                acc = acc + x * w_scalar

    acc = ct.astype(acc, output_flat.dtype)
    ct.store(output_flat, index=(bid,), tile=acc)


def run(input, kernel, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    cuTile 3D valid convolution — direct stencil matching Triton's method.
    input:  flat 1D tensor of size input_depth * input_rows * input_cols
    kernel: flat 1D tensor of size kernel_depth * kernel_rows * kernel_cols
    output: flat 1D tensor of size output_depth * output_rows * output_cols
    """
    global _last_autotune_config

    output_depth = input_depth - kernel_depth + 1
    output_rows_out = input_rows - kernel_rows + 1
    output_cols_out = input_cols - kernel_cols + 1
    total_out = output_depth * output_rows_out * output_cols_out

    if total_out <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(total_out, dtype=input.dtype, device=input.device)
    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (ct.cdiv(total_out, cfg.tile), 1, 1),
            kernel=_conv3d_stencil_kernel,
            args_fn=lambda cfg: (
                input, kernel, output,
                input_rows, input_cols,
                output_rows_out, output_cols_out,
                total_out,
                kernel_depth, kernel_rows, kernel_cols,
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
        grid = (ct.cdiv(total_out, cfg.tile), 1, 1)
        ct.launch(
            stream, grid, _conv3d_stencil_kernel,
            (input, kernel, output,
             input_rows, input_cols,
             output_rows_out, output_cols_out,
             total_out,
             kernel_depth, kernel_rows, kernel_cols,
             cfg.tile),
        )

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
