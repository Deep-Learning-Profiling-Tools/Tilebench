from types import SimpleNamespace

import torch
import cuda.tile as ct
from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [512, 1024, 2048]
    for occ in [4, 8]
]


@ct.kernel
def _touch_kernel(x, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(
        x,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ct.store(x, index=(bid,), tile=x_tile)


_tuner = CutileAutotuner(_touch_kernel)


def run(q, k, v, causal=True, autotune: bool = False, **kwargs):
    output = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=bool(causal),
    )

    if autotune:
        flat_output = output.reshape(-1)
        stream = torch.cuda.current_stream()
        cfg = _tuner.tune_or_cached(
            shape_key=(q.shape[0], q.shape[1], q.shape[2], q.shape[3], int(bool(causal))),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (1, 1, 1),
            args_fn=lambda cfg: (flat_output, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({"tile": cfg.tile, "occupancy": cfg.occupancy})
        kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
        ct.launch(stream, (1, 1, 1), kernel, (flat_output, cfg.tile))
    else:
        _last_autotune_config.clear()
        _last_autotune_config.update({"tile": 512, "occupancy": 4})

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
