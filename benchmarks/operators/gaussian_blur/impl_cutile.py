from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:  # pragma: no cover
    ct_experimental = None

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile=256, occupancy=2)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [128, 256, 512, 1024]
    for occ in [1, 2, 4]
]
_last_autotune_config = None


@ct.kernel
def _gaussian_blur_stencil_kernel(
    input_flat,
    kernel_flat,
    output_flat,
    input_rows,
    input_cols,
    total_elements,
    kernel_rows: ConstInt,
    kernel_cols: ConstInt,
    TILE: ConstInt,
):
    """
    Direct 2D stencil matching Triton's per-pixel-offset method:
      - Each block handles TILE consecutive output pixels (flat 1D layout).
      - For each (kr, kc), compute per-pixel input index and use ct.gather
        for runtime-computed loads (equivalent to Triton's pointer arithmetic).
      - ct.gather's padding_value handles zero-padded boundaries.
    """
    bid = ct.bid(0)
    offsets = bid * TILE + ct.arange(TILE, dtype=np.int32)
    mask = offsets < total_elements

    row = offsets // input_cols
    col = offsets % input_cols

    center_r = kernel_rows // 2
    center_c = kernel_cols // 2

    acc = ct.zeros((TILE,), dtype=np.float32)

    for kr in range(kernel_rows):  # compile-time unrolled
        for kc in range(kernel_cols):
            in_r = row + (kr - center_r)
            in_c = col + (kc - center_c)
            valid = mask & (in_r >= 0) & (in_r < input_rows) & (in_c >= 0) & (in_c < input_cols)

            # ct.where clamps invalid positions to -1; ct.gather treats negatives as OOB → 0.0
            input_idx = ct.where(valid, in_r * input_cols + in_c, -1)
            x = ct.gather(input_flat, input_idx, padding_value=0.0)
            x = ct.astype(x, np.float32)

            w_scalar = ct.load(kernel_flat, index=(kr * kernel_cols + kc,), shape=())
            w_scalar = ct.astype(w_scalar, np.float32)

            acc = acc + x * w_scalar

    # ct.store silently ignores OOB writes for the final partial tile.
    ct.store(output_flat, index=(bid,), tile=acc)


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    cuTile 2D Gaussian blur — direct stencil matching Triton's method.
    input:  flat 1D tensor of size input_rows * input_cols
    kernel: flat 1D tensor of size kernel_rows * kernel_cols
    output: flat 1D tensor of size input_rows * input_cols
    """
    global _last_autotune_config

    total_elements = input_rows * input_cols
    if total_elements <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    # Accumulator is fp32; allocate output in fp32 and cast to input.dtype on host
    # (matches the 3d_conv / conv2d_fwd cuTile pattern).
    output = torch.empty(total_elements, dtype=torch.float32, device=input.device)
    stream = torch.cuda.current_stream()

    input_f32 = input.float()
    kernel_f32 = kernel.float()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (ct.cdiv(total_elements, cfg.tile), 1, 1),
            kernel=_gaussian_blur_stencil_kernel,
            args_fn=lambda cfg: (
                input_f32, kernel_f32, output,
                input_rows, input_cols, total_elements,
                kernel_rows, kernel_cols,
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
        grid = (ct.cdiv(total_elements, cfg.tile), 1, 1)
        ct.launch(
            stream, grid, _gaussian_blur_stencil_kernel,
            (input_f32, kernel_f32, output,
             input_rows, input_cols, total_elements,
             kernel_rows, kernel_cols,
             cfg.tile),
        )

    return output.to(input.dtype)


def get_last_config() -> dict | None:
    return _last_autotune_config
