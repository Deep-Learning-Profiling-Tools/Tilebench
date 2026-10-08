from types import SimpleNamespace

import cuda.tile as ct
import torch

from tilebench.core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile_r=4, tile_c=128, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile_r=tr, tile_c=tc, occupancy=occ)
    for tr, tc in [(1, 128), (1, 256), (1, 512), (2, 128), (2, 256),
                   (4, 128), (4, 256), (8, 64)]
    for occ in [4, 8, 16]
]
_last_autotune_config: dict = {}


@ct.kernel
def max_pool2d_kernel(
    x3,
    out3,
    kernel_size: ConstInt,
    stride: ConstInt,
    padding: ConstInt,
    TILE_R: ConstInt,
    TILE_C: ConstInt,
):
    plane = ct.bid(0)
    bh = ct.bid(1)
    bw = ct.bid(2)

    oh2 = (bh * TILE_R + ct.arange(TILE_R, dtype=ct.int32))[:, None]
    ow2 = (bw * TILE_C + ct.arange(TILE_C, dtype=ct.int32))[None, :]

    acc = ct.full((TILE_R, TILE_C), -float("inf"), dtype=x3.dtype)

    for kh in range(kernel_size):
        for kw in range(kernel_size):
            ih = oh2 * stride + (kh - padding)
            iw = ow2 * stride + (kw - padding)
            x = ct.gather(x3, (plane, ih, iw), padding_value=-float("inf"))
            acc = ct.maximum(acc, x)

    ct.store(out3, index=(plane, bh, bw),
             tile=ct.reshape(acc, (1, TILE_R, TILE_C)))


_tuner = CutileAutotuner(max_pool2d_kernel)


def run(input, N, C, H, W, kernel_size, stride, padding,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    total_out = N * C * H_out * W_out

    if total_out <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(total_out, dtype=input.dtype, device=input.device)
    x3 = input.view(N * C, H, W)
    out3 = output.view(N * C, H_out, W_out)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(total_out, kernel_size, stride, padding, str(input.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (
                N * C,
                ct.cdiv(H_out, cfg.tile_r),
                ct.cdiv(W_out, cfg.tile_c),
            ),
            args_fn=lambda cfg: (
                x3, out3, kernel_size, stride, padding,
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
        N * C,
        ct.cdiv(H_out, cfg.tile_r),
        ct.cdiv(W_out, cfg.tile_c),
    )
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, grid, kernel,
              (x3, out3, kernel_size, stride, padding,
               cfg.tile_r, cfg.tile_c))

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
