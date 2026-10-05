from typing import Annotated

import torch
import cuda.tile as ct


FlatArray = Annotated[
    ct.Array,
    ct.ArrayAnnotation(static_shape_dims=(0,)),
]


@ct.kernel(num_ctas=1, occupancy=8, opt_level=3)
def _vector_add(x: FlatArray, y: FlatArray, out: FlatArray):
    bid = ct.bid(0)
    a = ct.load(
        x,
        (bid,),
        (8192,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=10,
        allow_tma=False,
    )
    b = ct.load(
        y,
        (bid,),
        (8192,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=10,
        allow_tma=False,
    )
    ct.store(
        out,
        (bid,),
        a + b,
        latency=10,
        allow_tma=False,
    )


def run(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    if n > 0:
        ct.launch(
            torch.cuda.current_stream(x.device),
            (ct.cdiv(n, 8192),),
            _vector_add,
            (x, y, out),
        )
    return out


def get_last_config() -> dict:
    return {
        "tile_size": 8192,
        "num_ctas": 1,
        "occupancy": 8,
        "opt_level": 3,
        "allow_tma": False,
        "load_latency": 10,
        "store_latency": 10,
        "static_shape_dims": (0,),
    }
