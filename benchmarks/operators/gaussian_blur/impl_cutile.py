from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile=256, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048]
    for occ in [4, 8, 16]
]
_last_autotune_config: dict = {}


@ct.kernel
def gaussian_blur_kernel(
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
    offsets = bid * TILE + ct.arange(TILE, dtype=ct.int32)
    mask = offsets < total_elements

    row = offsets // input_cols
    col = offsets % input_cols

    center_r = kernel_rows // 2
    center_c = kernel_cols // 2

    acc = ct.zeros((TILE,), dtype=ct.float32)

    for kr in range(kernel_rows):  # compile-time unrolled
        for kc in range(kernel_cols):
            in_r = row + (kr - center_r)
            in_c = col + (kc - center_c)
            valid = mask & (in_r >= 0) & (in_r < input_rows) & (in_c >= 0) & (in_c < input_cols)

            # ct.where clamps invalid positions to -1; ct.gather treats negatives as OOB → 0.0
            input_idx = ct.where(valid, in_r * input_cols + in_c, -1)
            x = ct.gather(input_flat, input_idx, padding_value=0.0)
            x = ct.astype(x, ct.float32)

            w_scalar = ct.load(kernel_flat, index=(kr * kernel_cols + kc,), shape=())
            w_scalar = ct.astype(w_scalar, ct.float32)

            acc = acc + x * w_scalar

    acc = ct.astype(acc, output_flat.dtype)
    ct.store(output_flat, index=(bid,), tile=acc)


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(gaussian_blur_kernel)


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    cuTile 2D Gaussian blur — direct stencil matching Triton's method.
    input:  flat 1D tensor of size input_rows * input_cols
    kernel: flat 1D tensor of size kernel_rows * kernel_cols
    output: flat 1D tensor of size input_rows * input_cols
    """

    total_elements = input_rows * input_cols
    if total_elements <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(total_elements, dtype=input.dtype, device=input.device)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(total_elements, kernel_rows, kernel_cols),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (ct.cdiv(total_elements, cfg.tile), 1, 1),
            args_fn=lambda cfg: (
                input, kernel, output,
                input_rows, input_cols, total_elements,
                kernel_rows, kernel_cols,
                cfg.tile,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tile":      cfg.tile,
            "occupancy": cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    grid = (ct.cdiv(total_elements, cfg.tile), 1, 1)
    kernel_obj = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream, grid, kernel_obj,
        (input, kernel, output,
         input_rows, input_cols, total_elements,
         kernel_rows, kernel_cols,
         cfg.tile),
    )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
