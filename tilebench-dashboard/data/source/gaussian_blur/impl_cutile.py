from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile_r=2, tile_c=128, occupancy=4)
_SEARCH_SPACE = [
    SimpleNamespace(tile_r=tr, tile_c=tc, occupancy=occ)
    for tr, tc in [(1, 128), (1, 256), (1, 512), (2, 128), (2, 256),
                   (4, 128), (4, 256), (8, 64)]
    for occ in [4, 8, 16]
]
_last_autotune_config: dict = {}


@ct.kernel
def gaussian_blur_kernel(
    x2d,
    kernel_flat,
    out2d,
    kernel_rows: ConstInt,
    kernel_cols: ConstInt,
    TILE_R: ConstInt,
    TILE_C: ConstInt,
):
    bh = ct.bid(0)
    bw = ct.bid(1)

    r2 = (bh * TILE_R + ct.arange(TILE_R, dtype=ct.int32))[:, None]
    c2 = (bw * TILE_C + ct.arange(TILE_C, dtype=ct.int32))[None, :]

    acc = ct.zeros((TILE_R, TILE_C), dtype=ct.float32)

    for kr in range(kernel_rows):
        for kc in range(kernel_cols):
            in_r = r2 + (kr - kernel_rows // 2)
            in_c = c2 + (kc - kernel_cols // 2)
            x = ct.gather(x2d, (in_r, in_c), padding_value=0.0)

            w_scalar = ct.load(kernel_flat,
                               index=(kr * kernel_cols + kc,), shape=())
            acc = acc + ct.astype(x, ct.float32) * ct.astype(w_scalar,
                                                             ct.float32)

    ct.store(out2d, index=(bh, bw), tile=ct.astype(acc, out2d.dtype))


_tuner = CutileAutotuner(gaussian_blur_kernel)


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    total_elements = input_rows * input_cols
    if total_elements <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(total_elements, dtype=input.dtype, device=input.device)
    x2d = input.view(input_rows, input_cols)
    out2d = output.view(input_rows, input_cols)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(total_elements, kernel_rows, kernel_cols, str(input.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (
                ct.cdiv(input_rows, cfg.tile_r),
                ct.cdiv(input_cols, cfg.tile_c),
                1,
            ),
            args_fn=lambda cfg: (
                x2d, kernel, out2d, kernel_rows, kernel_cols,
                cfg.tile_r, cfg.tile_c,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tile_r":    cfg.tile_r,
            "tile_c":    cfg.tile_c,
            "occupancy": cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    grid = (
        ct.cdiv(input_rows, cfg.tile_r),
        ct.cdiv(input_cols, cfg.tile_c),
        1,
    )
    kernel_obj = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, grid, kernel_obj,
              (x2d, kernel, out2d, kernel_rows, kernel_cols,
               cfg.tile_r, cfg.tile_c))

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
