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
    offsets = ct.bid(0) * 8192 + ct.arange(8192, dtype=ct.int32)

    x_memory = x.get_raw_memory()
    y_memory = y.get_raw_memory()
    out_memory = out.get_raw_memory()

    a = x_memory.load_offset(offsets)
    b = y_memory.load_offset(offsets)
    out_memory.store_offset(offsets, a + b)


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
        "static_shape_dims": (0,),
        "memory_access": "raw",
        "masked": False,
    }
