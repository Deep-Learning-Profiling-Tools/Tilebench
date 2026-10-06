from typing import Annotated

import torch
import cuda.tile as ct


FlatArray = Annotated[
    ct.Array,
    ct.ArrayAnnotation(static_shape_dims=(0,)),
]


@ct.kernel(
    num_ctas=1,
    occupancy=12,
    opt_level=3,
    num_worker_warps=4,
)
def _vector_add(x: FlatArray, y: FlatArray, out: FlatArray):
    base = ct.bid(0) * 2
    for part in ct.static_iter(range(2)):
        index = base + part
        a = ct.load(
            x,
            (index,),
            (4096,),
            padding_mode=ct.PaddingMode.ZERO,
            latency=4,
            allow_tma=False,
        )
        b = ct.load(
            y,
            (index,),
            (4096,),
            padding_mode=ct.PaddingMode.ZERO,
            latency=4,
            allow_tma=False,
        )
        ct.store(out, (index,), a + b, allow_tma=False)


def run(x, y):
    out = torch.empty_like(x)
    ct.launch(
        torch.cuda.current_stream(x.device),
        (2560,),
        _vector_add,
        (x, y, out),
    )
    return out


def get_last_config() -> dict:
    return {
        "n": 20971520,
        "tile_size": 4096,
        "tiles_per_block": 2,
        "grid_size": 2560,
        "num_ctas": 1,
        "occupancy": 12,
        "opt_level": 3,
        "num_worker_warps": 4,
        "load_latency": 4,
        "allow_tma": False,
        "static_shape_dims": (0,),
    }
