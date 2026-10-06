from typing import Annotated

import torch
import cuda.tile as ct


FlatArray = Annotated[
    ct.Array,
    ct.ArrayAnnotation(static_shape_dims=(0,)),
]


@ct.kernel(
    num_ctas=1,
    occupancy=6,
    opt_level=3,
    num_worker_warps=4,
)
def _vector_add(x: FlatArray, y: FlatArray, out: FlatArray):
    block = ct.bid(0)
    a = ct.load(
        x,
        (block,),
        (8192,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    b = ct.load(
        y,
        (block,),
        (8192,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    ct.store(out, (block,), a + b, allow_tma=False)


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
        "tile_size": 8192,
        "grid_size": 2560,
        "num_ctas": 1,
        "occupancy": 6,
        "opt_level": 3,
        "num_worker_warps": 4,
        "allow_tma": False,
        "static_shape_dims": (0,),
    }
