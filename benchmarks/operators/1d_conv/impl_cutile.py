from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048]
    for occ in [4, 8, 16]
]
_last_autotune_config: dict = {}


@ct.kernel
def conv1d_kernel(
    input_flat,
    kernel_flat,
    output_flat,
    kernel_size: ConstInt,
    TILE: ConstInt,
):
    bid = ct.bid(0)
    offsets = bid * TILE + ct.arange(TILE, dtype=ct.int32)

    acc = ct.zeros((TILE,), dtype=ct.float32)

    for j in range(kernel_size):  # compile-time unrolled
        input_idx = offsets + j
        x = ct.gather(input_flat, input_idx, padding_value=0.0)
        x = ct.astype(x, ct.float32)

        w_scalar = ct.load(kernel_flat, index=(j,), shape=())
        w_scalar = ct.astype(w_scalar, ct.float32)

        acc = acc + x * w_scalar

    ct.store(output_flat, index=(bid,), tile=ct.astype(acc, output_flat.dtype))


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(conv1d_kernel)


def run(input, kernel, input_size, kernel_size,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    cuTile 1D valid convolution — direct stencil matching Triton's method.
    input:  flat 1D tensor of size input_size
    kernel: flat 1D tensor of size kernel_size
    output: flat 1D tensor of size input_size - kernel_size + 1
    """

    output_size = input_size - kernel_size + 1
    if output_size <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(input_size, kernel_size, input.dtype),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (ct.cdiv(output_size, cfg.tile), 1, 1),
            args_fn=lambda cfg: (
                input, kernel, output,
                kernel_size,
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

    grid = (ct.cdiv(output_size, cfg.tile), 1, 1)
    kernel_obj = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream, grid, kernel_obj,
        (input, kernel, output, kernel_size, cfg.tile),
    )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
