"""cuTile single-channel VALID 2D correlation — direct stencil, 1D-flat tiling.

Mirrors 1d_conv / 3d_conv (NOT a 2D block tile): each CTA owns TILE *consecutive
flattened* output positions, decodes them back to (oh, ow), and for every kernel
tap (i, j) gathers input[oh+i, ow+j] with a **1D** index. Consecutive lanes read
consecutive input elements, so cuTile's ct.gather coalesces — unlike the previous
2D-tile version whose 2D (outer-product) gather index did not coalesce and
saturated L1 (~6x the sectors, ~5x slower than Triton).

Valid convolution ⇒ every in-bounds output position only touches in-bounds input
(oh+i <= input_rows-1, ow+j <= input_cols-1), so no input mask is needed:
ct.gather's padding handles the padded tail lanes and ct.store drops the OOB
writes of the last (partial) tile. Same method/precision as Triton (scalar weight
* shifted input, fp32 accumulate) — only the tiling differs.
"""
from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048, 4096]
    for occ in [4, 8, 16]
]


@ct.kernel
def conv2d_kernel(
    input_flat,     # 1D view of input:  [input_rows, input_cols]
    kernel_flat,    # 1D view of kernel: [kernel_rows, kernel_cols], contiguous
    output_flat,    # 1D view of output: [out_rows, out_cols]
    input_cols,
    output_cols,
    kernel_rows: ConstInt, kernel_cols: ConstInt,
    TILE: ConstInt,
):
    bid = ct.bid(0)
    offsets = bid * TILE + ct.arange(TILE, dtype=ct.int32)   # [TILE] flat output positions

    # Decode flat output offset -> (oh, ow)
    oh = offsets // output_cols
    ow = offsets % output_cols

    acc = ct.zeros((TILE,), dtype=ct.float32)

    for i in range(kernel_rows):          # compile-time unrolled
        for j in range(kernel_cols):
            # input[oh+i, ow+j] as a 1D flat index (contiguous NCHW: row stride = input_cols)
            input_idx = (oh + i) * input_cols + (ow + j)   # [TILE], 1D -> coalesced
            x = ct.gather(input_flat, input_idx, padding_value=0.0)
            x = ct.astype(x, ct.float32)

            w = ct.load(kernel_flat, index=(i * kernel_cols + j,), shape=())
            w = ct.astype(w, ct.float32)

            acc = acc + x * w

    acc = ct.astype(acc, output_flat.dtype)
    ct.store(output_flat, index=(bid,), tile=acc)


_tuner = CutileAutotuner(conv2d_kernel)


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_rows: int, input_cols: int,
        kernel_rows: int, kernel_cols: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    cuTile single-channel VALID 2D correlation — direct stencil (1D-flat tiling).

    input:  [input_rows, input_cols]
    kernel: [kernel_rows, kernel_cols]
    output: [input_rows - kernel_rows + 1, input_cols - kernel_cols + 1]
    """
    assert input.is_contiguous() and kernel.is_contiguous()
    out_rows = input_rows - kernel_rows + 1
    out_cols = input_cols - kernel_cols + 1
    total_out = out_rows * out_cols

    output = torch.empty((out_rows, out_cols), device=input.device, dtype=input.dtype)
    input_flat = input.view(-1)
    kernel_flat = kernel.view(-1)
    output_flat = output.view(-1)

    stream = torch.cuda.current_stream()

    base = (input_flat, kernel_flat, output_flat, input_cols, out_cols,
            kernel_rows, kernel_cols)

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(input_rows, input_cols, kernel_rows, kernel_cols, input.dtype),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (ct.cdiv(total_out, cfg.tile), 1, 1),
            args_fn=lambda cfg: base + (cfg.tile,),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tile": cfg.tile,
            "occupancy": cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    grid = (ct.cdiv(total_out, cfg.tile), 1, 1)
    kernel_obj = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, grid, kernel_obj, base + (cfg.tile,))

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
